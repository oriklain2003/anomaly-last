import React from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

/**
 * Comprehensive markdown renderer for chat messages
 * Supports: # headers, **bold**, *italic*, - lists, numbered lists, 
 * --- dividers, `code`, > blockquotes, [links](url)
 */
export function renderMarkdown(text: string): React.ReactNode {
    const lines = text.split('\n');
    const elements: React.ReactNode[] = [];
    let inBlockquote = false;
    let blockquoteLines: string[] = [];
    
    const flushBlockquote = (keyPrefix: string) => {
        if (blockquoteLines.length > 0) {
            elements.push(
                <div 
                    key={`${keyPrefix}-blockquote`} 
                    className="border-l-2 border-primary/50 pl-3 py-1 my-2 bg-primary/5 rounded-r text-white/80 italic"
                >
                    {blockquoteLines.map((line, i) => (
                        <div key={i}>{renderInlineMarkdown(line)}</div>
                    ))}
                </div>
            );
            blockquoteLines = [];
        }
        inBlockquote = false;
    };
    
    lines.forEach((line, lineIndex) => {
        // Blockquote handling
        if (line.trim().startsWith('>')) {
            inBlockquote = true;
            blockquoteLines.push(line.trim().replace(/^>\s*/, ''));
            return;
        } else if (inBlockquote) {
            flushBlockquote(`line-${lineIndex}`);
        }
        
        // Horizontal rule (---, ***, ___, or more)
        if (/^[\s]*[-*_]{3,}[\s]*$/.test(line)) {
            elements.push(
                <hr key={lineIndex} className="border-white/20 my-3" />
            );
            return;
        }
        
        // Empty line
        if (line.trim() === '') {
            elements.push(<div key={lineIndex} className="h-2" />);
            return;
        }
        
        // Headers (# ## ### #### etc)
        const headerMatch = line.match(/^(#{1,6})\s+(.+)$/);
        if (headerMatch) {
            const level = headerMatch[1].length;
            const content = headerMatch[2];
            const headerStyles: Record<number, string> = {
                1: 'text-lg font-bold text-white mt-3 mb-2',
                2: 'text-base font-bold text-white mt-2 mb-1.5',
                3: 'text-sm font-bold text-white/90 mt-2 mb-1',
                4: 'text-sm font-semibold text-white/80 mt-1.5 mb-1',
                5: 'text-xs font-semibold text-white/70 mt-1 mb-0.5',
                6: 'text-xs font-medium text-white/60 mt-1 mb-0.5',
            };
            elements.push(
                <div key={lineIndex} className={headerStyles[level] || headerStyles[3]}>
                    {renderInlineMarkdown(content)}
                </div>
            );
            return;
        }
        
        // Unordered list items (-, *, •, ◦, ▪)
        const listMatch = line.match(/^(\s*)([-•*◦▪])\s+(.+)$/);
        if (listMatch) {
            const [, indent, , content] = listMatch;
            const indentLevel = Math.floor(indent.length / 2);
            const bulletStyle = indentLevel > 0 ? 'text-white/40' : 'text-primary';
            elements.push(
                <div 
                    key={lineIndex} 
                    className="flex items-start gap-2 py-0.5"
                    style={{ paddingLeft: `${indentLevel * 16}px` }}
                >
                    <span className={`${bulletStyle} shrink-0 mt-0.5 text-xs`}>•</span>
                    <span className="flex-1">{renderInlineMarkdown(content)}</span>
                </div>
            );
            return;
        }
        
        // Numbered/lettered list (1. 2. a. b. etc)
        const numberedMatch = line.match(/^(\s*)(\d+|[a-zA-Z])[\.\)]\s+(.+)$/);
        if (numberedMatch) {
            const [, indent, marker, content] = numberedMatch;
            const indentLevel = Math.floor(indent.length / 2);
            elements.push(
                <div 
                    key={lineIndex} 
                    className="flex items-start gap-2 py-0.5"
                    style={{ paddingLeft: `${indentLevel * 16}px` }}
                >
                    <span className="text-white/50 shrink-0 min-w-[1.5rem] text-right">{marker}.</span>
                    <span className="flex-1">{renderInlineMarkdown(content)}</span>
                </div>
            );
            return;
        }
        
        // Task list [ ] or [x]
        const taskMatch = line.match(/^(\s*)[-*]\s+\[([ xX])\]\s+(.+)$/);
        if (taskMatch) {
            const [, indent, checked, content] = taskMatch;
            const indentLevel = Math.floor(indent.length / 2);
            const isChecked = checked.toLowerCase() === 'x';
            elements.push(
                <div 
                    key={lineIndex} 
                    className="flex items-start gap-2 py-0.5"
                    style={{ paddingLeft: `${indentLevel * 16}px` }}
                >
                    <span className={`shrink-0 mt-0.5 w-4 h-4 rounded border flex items-center justify-center text-xs ${
                        isChecked ? 'bg-primary/20 border-primary text-primary' : 'border-white/30'
                    }`}>
                        {isChecked && '✓'}
                    </span>
                    <span className={`flex-1 ${isChecked ? 'line-through text-white/50' : ''}`}>
                        {renderInlineMarkdown(content)}
                    </span>
                </div>
            );
            return;
        }
        
        // Regular line with inline formatting
        elements.push(
            <div key={lineIndex} className="py-0.5">
                {renderInlineMarkdown(line)}
            </div>
        );
    });
    
    // Flush any remaining blockquote
    flushBlockquote('final');
    
    return <>{elements}</>;
}

function extractThinkingBlocks(raw: string): { cleanText: string; thinkingBlocks: string[] } {
    const thinkingBlocks: string[] = [];
    const re = /<thinking>([\s\S]*?)<\/thinking>/gi;

    let m: RegExpExecArray | null;
    while ((m = re.exec(raw)) !== null) {
        const block = (m[1] ?? '').trim();
        if (block) thinkingBlocks.push(block);
    }

    const cleanText = raw
        .replace(re, '')
        .replace(/\n{3,}/g, '\n\n')
        .trim();

    return { cleanText, thinkingBlocks };
}

const ThinkingDisclosure: React.FC<{ blocks: string[] }> = ({ blocks }) => {
    const [open, setOpen] = React.useState(false);
    const [revealedLines, setRevealedLines] = React.useState(0);

    const fullText = React.useMemo(() => blocks.join('\n\n---\n\n').trim(), [blocks]);
    const lines = React.useMemo(() => fullText.split('\n'), [fullText]);

    React.useEffect(() => {
        if (!open) return;
        if (revealedLines >= lines.length) return;

        // "Sleep per line" reveal: small delay so it feels like real step-by-step thinking.
        const delayMs = lines.length > 80 ? 35 : lines.length > 40 ? 65 : 120;
        const t = window.setTimeout(() => {
            setRevealedLines((n) => Math.min(n + 1, lines.length));
        }, delayMs);

        return () => window.clearTimeout(t);
    }, [open, revealedLines, lines.length]);

    const shownText = open ? lines.slice(0, Math.max(1, revealedLines)).join('\n') : '';

    return (
        <div className="mt-3 border-t border-white/10 pt-2">
            <button
                type="button"
                onClick={() => setOpen((v) => !v)}
                aria-expanded={open}
                className="flex items-center gap-2 text-xs text-white/60 hover:text-white/80 transition-colors"
            >
                {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                <span className="font-medium">Thinking</span>
                {!open && <span className="text-white/35">(hidden)</span>}
            </button>

            {open && (
                <div className="mt-2 rounded-lg bg-black/20 border border-white/10 p-3 text-xs text-white/75">
                    <div className="markdown-content">{renderMarkdown(shownText)}</div>
                </div>
            )}
        </div>
    );
};

/**
 * Render inline markdown: **bold**, *italic*, `code`, ~~strikethrough~~, [links](url)
 */
function renderInlineMarkdown(text: string): React.ReactNode {
    const parts: React.ReactNode[] = [];
    let remaining = text;
    let keyIndex = 0;
    
    // Process inline formatting patterns (order matters - most specific first)
    const patterns: Array<{
        regex: RegExp;
        render: (match: string, extra?: string) => React.ReactNode;
    }> = [
        // Links [text](url)
        {
            regex: /\[([^\]]+)\]\(([^)]+)\)/,
            render: (text: string, url?: string) => (
                <a 
                    key={keyIndex++} 
                    href={url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-primary hover:underline"
                >
                    {text}
                </a>
            )
        },
        // Bold + Italic ***text***
        {
            regex: /\*\*\*(.+?)\*\*\*/,
            render: (match: string) => (
                <strong key={keyIndex++} className="font-bold italic text-white">{match}</strong>
            )
        },
        // Bold **text** or __text__
        {
            regex: /\*\*(.+?)\*\*|__(.+?)__/,
            render: (match: string) => (
                <strong key={keyIndex++} className="font-bold text-white">{match}</strong>
            )
        },
        // Strikethrough ~~text~~
        {
            regex: /~~(.+?)~~/,
            render: (match: string) => (
                <span key={keyIndex++} className="line-through text-white/50">{match}</span>
            )
        },
        // Italic *text* or _text_ (but not inside words like file_name)
        {
            regex: /(?<![a-zA-Z0-9_])\*([^*\n]+)\*(?![a-zA-Z0-9_])|(?<![a-zA-Z0-9])\b_([^_\n]+)_\b(?![a-zA-Z0-9])/,
            render: (match: string) => (
                <em key={keyIndex++} className="italic text-white/90">{match}</em>
            )
        },
        // Inline code `text`
        {
            regex: /`([^`]+)`/,
            render: (match: string) => (
                <code key={keyIndex++} className="px-1.5 py-0.5 rounded bg-white/10 text-primary font-mono text-xs">
                    {match}
                </code>
            )
        },
    ];
    
    while (remaining.length > 0) {
        let earliestMatch: { 
            index: number; 
            length: number; 
            content: string; 
            extra?: string;
            render: (m: string, e?: string) => React.ReactNode 
        } | null = null;
        
        for (const pattern of patterns) {
            const match = remaining.match(pattern.regex);
            if (match && match.index !== undefined) {
                if (earliestMatch === null || match.index < earliestMatch.index) {
                    // For links, we need both the text and URL
                    const content = match[1] || match[2] || '';
                    const extra = pattern.regex.toString().includes('\\]\\(') ? match[2] : undefined;
                    earliestMatch = {
                        index: match.index,
                        length: match[0].length,
                        content: content,
                        extra: extra,
                        render: pattern.render
                    };
                }
            }
        }
        
        if (earliestMatch) {
            // Add text before the match
            if (earliestMatch.index > 0) {
                parts.push(remaining.slice(0, earliestMatch.index));
            }
            // Add the formatted element
            parts.push(earliestMatch.render(earliestMatch.content, earliestMatch.extra));
            // Continue with the rest
            remaining = remaining.slice(earliestMatch.index + earliestMatch.length);
        } else {
            // No more matches, add remaining text
            parts.push(remaining);
            break;
        }
    }
    
    return <>{parts}</>;
}

/**
 * TypewriterText with Markdown support
 */
interface TypewriterMarkdownProps {
    text: string;
    shouldAnimate: boolean;
}

export const TypewriterMarkdown: React.FC<TypewriterMarkdownProps> = ({ text, shouldAnimate }) => {
    const { cleanText, thinkingBlocks } = React.useMemo(() => extractThinkingBlocks(text), [text]);
    const [displayedText, setDisplayedText] = React.useState(shouldAnimate ? '' : cleanText);
    const hasAnimatedRef = React.useRef(false);

    React.useEffect(() => {
        if (!shouldAnimate) {
            setDisplayedText(cleanText);
            return;
        }
        
        if (hasAnimatedRef.current) return;

        setDisplayedText('');
        let i = 0;
        const timer = setInterval(() => {
            setDisplayedText(cleanText.slice(0, i + 1));
            i++;
            if (i >= cleanText.length) {
                clearInterval(timer);
                hasAnimatedRef.current = true;
            }
        }, 10); // Slightly faster for better UX

        return () => clearInterval(timer);
    }, [cleanText, shouldAnimate]);

    return (
        <div>
            <div className="markdown-content">{renderMarkdown(displayedText)}</div>
            {thinkingBlocks.length > 0 && <ThinkingDisclosure blocks={thinkingBlocks} />}
        </div>
    );
};
