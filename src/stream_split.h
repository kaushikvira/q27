// Streaming splitter for Qwopus output markers: <think>...</think> reasoning
// and <tool_call>...</tool_call> function calls, both emitted as plain-text
// tokens. Feeds token-by-token decoded text and routes segments into THINK /
// TEXT / TOOL channels, holding back any tail that could be a partial marker.
// Markers do not nest; tool_calls can appear only outside think blocks in
// well-formed output, but we tolerate them inside by scanning TEXT only.
// A closed tool followed by another structural channel with no intervening
// payload emits an empty non-TOOL boundary segment. Consumers buffer one TOOL
// segment at a time and flush on any non-TOOL segment, so without the boundary
// adjacent calls fold into one buffer. An empty think block must also preserve
// this separation. Empty boundary segments are no-ops for consumers that do
// not buffer tools.
#pragma once
#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "markdown_lex.h"

namespace q27 {

struct StreamSplitter {
    enum Chan { TEXT, THINK, TOOL };
    Chan chan = TEXT;
    std::string hold;
    // Set when a TOOL closer returned us to TEXT and no non-TOOL segment has
    // been emitted since. The next structural opener may need an empty segment
    // to flush consumers' pending tool buffer.
    bool tool_boundary = false;
    // display() enables the unterminated-string-ends-at-newline rule: this
    // state only ever answers "is this marker structural?", never "where does
    // a rescued call begin?"
    JsonStringLexState text_string_state=JsonStringLexState::display();
    // Are we inside a <parameter=KEY>...</parameter> VALUE in the TOOL channel?
    // Advanced over every TOOL byte as it leaves `hold`, because the decision
    // below needs state the emitted bytes have already carried away.
    bool tool_in_value=false;
    MarkdownFenceLexState text_markdown_state;

    static constexpr const char* T_OPEN = "<think>";
    static constexpr const char* T_CLOSE = "</think>";
    static constexpr const char* C_OPEN = "<tool_call>";
    static constexpr const char* C_CLOSE = "</tool_call>";
    static constexpr const char* P_OPEN = "<parameter=";
    static constexpr const char* P_CLOSE = "</parameter>";
    static constexpr const char* F_CLOSE = "</function>";

    // Advance tool_in_value over TOOL bytes that are about to be emitted.
    void advance_tool_value(const std::string& s) {
        size_t i=0;
        while(i<s.size()) {
            if(!tool_in_value) {
                size_t a=s.find(P_OPEN,i);
                if(a==std::string::npos) return;
                size_t gt=s.find('>',a+strlen(P_OPEN));
                if(gt==std::string::npos) return;  // opener split across feeds
                tool_in_value=true; i=gt+1;
            } else {
                size_t e=s.find(P_CLOSE,i), f=s.find(F_CLOSE,i);
                size_t end=std::min(e,f);
                if(end==std::string::npos) return;
                tool_in_value=false;
                i=end+(end==e?strlen(P_CLOSE):strlen(F_CLOSE));
            }
        }
    }

    // Find the STRUCTURAL </tool_call> in `h`, i.e. one that is not sitting
    // inside a parameter value. Returns npos if there is none yet, and sets
    // *pending to the first </tool_call> inside a value that has NOT closed --
    // a PROVISIONAL closer.
    //
    // The XML dialect has no escaping, so a call that legitimately writes about
    // the protocol (a doc, a verifier, this BUILDLOG) puts a literal
    // </tool_call> inside <parameter=content>. Closing the channel at that byte
    // truncated the call, lost it entirely, and leaked the trailing
    // </parameter></function> as visible text.
    //
    // But a model that FORGETS </parameter> and closes with </tool_call> is a
    // real, working case today (api_common tolerates the missing final
    // </parameter> at EOF). So an in-value </tool_call> cannot simply be
    // ignored: it is provisional. If the value later closes, it was content;
    // if generation ends first, it was the closer after all. flush() resolves
    // it that way, and until then those bytes are held rather than emitted, so
    // the choice is still open when the answer arrives.
    size_t structural_tool_close(const std::string& h, bool in_value,
                                 size_t* pending) const {
        *pending=std::string::npos;
        size_t i=0;
        while(i<h.size()) {
            if(!in_value) {
                size_t a=h.find(P_OPEN,i), b=h.find(C_CLOSE,i);
                if(b!=std::string::npos && (a==std::string::npos || b<a)) return b;
                if(a==std::string::npos) return std::string::npos;
                size_t gt=h.find('>',a+strlen(P_OPEN));
                if(gt==std::string::npos) return std::string::npos;
                in_value=true; i=gt+1;
            } else {
                size_t e=h.find(P_CLOSE,i), f=h.find(F_CLOSE,i), c=h.find(C_CLOSE,i);
                size_t end=std::min(e,f);
                if(c!=std::string::npos && c<end) {
                    if(*pending==std::string::npos) *pending=c;
                    i=c+strlen(C_CLOSE);
                    continue;
                }
                if(end==std::string::npos) return std::string::npos; // value open
                *pending=std::string::npos; // value closed -> those were content
                in_value=false;
                i=end+(end==e?strlen(P_CLOSE):strlen(F_CLOSE));
            }
        }
        return std::string::npos;
    }

    std::vector<std::pair<Chan, std::string>> feed(const std::string& piece) {
        hold += piece;
        std::vector<std::pair<Chan, std::string>> out;
        for (;;) {
            if (chan == TEXT) {
                // Only markers in executable model text are structural.
                // Markdown/HTML code, quotes, list examples, and JSON strings
                // remain visible bytes; treating those as TOOL output would
                // let echoed or retrieved documentation trigger a client call.
                size_t pt = hold.find(T_OPEN), pc = hold.find(C_OPEN),
                       sc = hold.find(C_CLOSE);
                size_t e = std::min(pt, std::min(pc, sc));
                if (e != std::string::npos) {
                    const char* marker=e==pt?T_OPEN:e==pc?C_OPEN:C_CLOSE;
                    emit_text_head(out,e);
                    if(e) tool_boundary=false;
                    text_string_state.settle_pending(marker[0]);
                    const bool structural=!text_string_state.in_inert_container() &&
                        display_text_context_is_executable(
                            hold,0,text_string_state,text_markdown_state,false);
                    if(structural) text_string_state.discard_pending_containers();
                    if (!structural) {
                        emit_text_head(out,strlen(marker));
                        tool_boundary=false;
                        continue;
                    }
                    if (e == pt) {
                        hold.erase(0, strlen(T_OPEN));
                        chan = THINK;
                        continue;
                    }
                    if (e == pc) {
                        if (tool_boundary) out.push_back({TEXT, ""});
                        tool_boundary = false;
                        hold.erase(0, strlen(C_OPEN));
                        chan = TOOL;
                        continue;
                    }
                    hold.erase(0, strlen(C_CLOSE)); // executable stray close
                    continue;
                }
                // hold back the longest suffix that prefixes any marker
                size_t keep = tail_keep(T_OPEN);
                keep = std::max(keep, tail_keep(C_OPEN));
                keep = std::max(keep, tail_keep(C_CLOSE));
                if (emit_head(out, keep)) tool_boundary = false;
                break;
            }
            const char* closer = chan == THINK ? T_CLOSE : C_CLOSE;
            size_t pending = std::string::npos;
            size_t p = chan == TOOL
                          ? structural_tool_close(hold, tool_in_value, &pending)
                          : hold.find(closer);
            if (chan == TOOL && p == std::string::npos &&
                pending != std::string::npos) {
                // a provisional closer is open: emit only the bytes before it
                // and hold the rest until the value closes (content) or the
                // generation ends (flush() promotes it to the real closer)
                if (pending) {
                    std::string head = hold.substr(0, pending);
                    advance_tool_value(head);
                    out.push_back({TOOL, std::move(head)});
                    hold.erase(0, pending);
                    tool_boundary = false;
                }
                break;
            }
            if (p != std::string::npos) {
                if (p > 0) {
                    out.push_back({chan, hold.substr(0, p)});
                    tool_boundary = false;
                } else if (chan == THINK && tool_boundary) {
                    out.push_back({THINK, ""});
                }
                hold.erase(0, p + strlen(closer));
                tool_boundary = (chan == TOOL);
                if (chan == TOOL) tool_in_value = false;
                chan = TEXT;
                continue;
            }
            if (emit_head(out, chan == TOOL ? tool_keep() : tail_keep(closer)))
                tool_boundary = false;
            break;
        }
        return out;
    }

    std::vector<std::pair<Chan, std::string>> flush() {
        std::vector<std::pair<Chan, std::string>> out;
        if (chan == TOOL && !hold.empty()) {
            // Resolve a provisional closer: the generation ended without the
            // value closing, so that in-value </tool_call> WAS the closer (the
            // model dropped its </parameter>). Split there, exactly as the
            // pre-existing behaviour did, so that working case is preserved.
            size_t pending=std::string::npos;
            if (structural_tool_close(hold,tool_in_value,&pending)==std::string::npos &&
                pending!=std::string::npos) {
                if (pending) out.push_back({TOOL, hold.substr(0,pending)});
                std::string rest=hold.substr(pending+strlen(C_CLOSE));
                hold.clear();
                tool_in_value=false;
                chan=TEXT;
                if (!rest.empty()) {
                    consume_display_text_context(
                        text_string_state,text_markdown_state,rest);
                    out.push_back({TEXT,std::move(rest)});
                }
                tool_boundary=false;
                return out;
            }
        }
        if (!hold.empty()) {
            if (chan == TEXT)
                consume_display_text_context(
                    text_string_state,text_markdown_state,hold);
            out.push_back({chan, hold});
        } else if (chan == THINK && tool_boundary) out.push_back({THINK, ""});
        hold.clear();
        tool_boundary = false;
        return out;
    }

  private:
    void emit_text_head(std::vector<std::pair<Chan, std::string>>& out,
                        size_t count) {
        if (!count) return;
        std::string text=hold.substr(0,count);
        consume_display_text_context(
            text_string_state,text_markdown_state,text);
        out.push_back({TEXT,std::move(text)});
        hold.erase(0,count);
    }

    // How many trailing TOOL bytes must stay in `hold` for the value-state
    // machine to stay correct across feed() boundaries. The TEXT channel does
    // the same thing for its markers; the TOOL channel used to hold back only a
    // partial </tool_call>, so a <parameter= split across two tokens was never
    // recognised, tool_in_value stayed false, and the next </tool_call> inside
    // that value read as structural -- the bug, but only when streaming.
    size_t tool_keep() const {
        size_t k = tail_keep(C_CLOSE);
        k = std::max(k, tail_keep(P_OPEN));
        k = std::max(k, tail_keep(P_CLOSE));
        k = std::max(k, tail_keep(F_CLOSE));
        if (!tool_in_value) {
            // an opener whose '>' has not arrived yet: hold from the opener, or
            // advance_tool_value() would consume it without flipping the flag
            size_t a = hold.rfind(P_OPEN);
            if (a != std::string::npos &&
                hold.find('>', a + strlen(P_OPEN)) == std::string::npos)
                k = std::max(k, hold.size() - a);
        }
        return k;
    }
    size_t tail_keep(const char* marker) const {
        size_t mlen = strlen(marker);
        size_t maxk = std::min(hold.size(), mlen - 1);
        for (size_t k = maxk; k > 0; k--)
            if (hold.compare(hold.size() - k, k, marker, k) == 0) return k;
        return 0;
    }
    bool emit_head(std::vector<std::pair<Chan, std::string>>& out, size_t keep) {
        if (hold.size() <= keep) return false;
        const size_t count=hold.size()-keep;
        if (chan == TEXT) emit_text_head(out,count);
        else {
            std::string seg=hold.substr(0,count);
            if (chan == TOOL) advance_tool_value(seg);
            out.push_back({chan, std::move(seg)});
            hold.erase(0,count);
        }
        return true;
    }
};


} // namespace q27
