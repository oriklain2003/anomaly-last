import React, { useEffect, useMemo, useState } from 'react';
import { X, RotateCcw, Moon, Sun, ChevronDown } from 'lucide-react';
import { applyTheme, DEFAULT_THEME, loadSavedTheme, PRESET_THEMES, ThemeConfig, ThemePreset } from '../theme';

interface SettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
}

const humanizePresetName = (key: string) =>
    key.replace(/([A-Z])/g, ' $1').replace(/^./, (str) => str.toUpperCase());

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose }) => {
    const [theme, setTheme] = useState<ThemeConfig>(DEFAULT_THEME);
    const [darkExpanded, setDarkExpanded] = useState(false);
    const [lightExpanded, setLightExpanded] = useState(false);

    // Hydrate with saved colors on open
    useEffect(() => {
        if (!isOpen) return;
        const saved = loadSavedTheme();
        setTheme(saved);
        applyTheme(saved);
    }, [isOpen]);

    // Live preview while adjusting
    useEffect(() => {
        if (!isOpen) return;
        applyTheme(theme);
    }, [theme, isOpen]);

    const handleReset = () => {
        setTheme(DEFAULT_THEME);
        applyTheme(DEFAULT_THEME);
    };

    const { darkPresets, lightPresets } = useMemo(() => {
        const entries = Object.entries(PRESET_THEMES) as [string, ThemePreset][];
        return {
            darkPresets: entries.filter(([, preset]) => preset.category === 'dark'),
            lightPresets: entries.filter(([, preset]) => preset.category === 'light'),
        };
    }, []);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="bg-surface border border-white/10 rounded-xl p-6 w-[480px] max-h-[90vh] overflow-hidden shadow-2xl relative flex flex-col">
                <div className="flex justify-between items-center mb-6">
                    <h2 className="text-xl font-bold text-white">Theme Settings</h2>
                    <button onClick={onClose} className="text-white/60 hover:text-white transition-colors">
                        <X className="size-6" />
                    </button>
                </div>

                <div className="space-y-6 overflow-y-auto flex-1 pr-1">
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-white/80">Primary color (app background)</label>
                        <div className="flex gap-3">
                            <input
                                type="color"
                                value={theme.background}
                                onChange={(e) => setTheme((cur) => ({ ...cur, background: e.target.value }))}
                                className="h-10 w-20 rounded cursor-pointer bg-transparent border border-white/20 p-1"
                            />
                            <input
                                type="text"
                                value={theme.background}
                                onChange={(e) => setTheme((cur) => ({ ...cur, background: e.target.value }))}
                                className="flex-1 bg-background-dark border border-white/10 rounded px-3 text-white text-sm"
                            />
                        </div>
                    </div>

                    <div className="space-y-2">
                        <label className="text-sm font-medium text-white/80">Secondary color (items & cards)</label>
                        <div className="flex gap-3">
                            <input
                                type="color"
                                value={theme.surface}
                                onChange={(e) => setTheme((cur) => ({ ...cur, surface: e.target.value }))}
                                className="h-10 w-20 rounded cursor-pointer bg-transparent border border-white/20 p-1"
                            />
                            <input
                                type="text"
                                value={theme.surface}
                                onChange={(e) => setTheme((cur) => ({ ...cur, surface: e.target.value }))}
                                className="flex-1 bg-background-dark border border-white/10 rounded px-3 text-white text-sm"
                            />
                        </div>
                    </div>

                    <div className="space-y-2">
                        <label className="text-sm font-medium text-white/80">Third color (tabs & highlights)</label>
                        <div className="flex gap-3">
                            <input
                                type="color"
                                value={theme.accent}
                                onChange={(e) => setTheme((cur) => ({ ...cur, accent: e.target.value }))}
                                className="h-10 w-20 rounded cursor-pointer bg-transparent border border-white/20 p-1"
                            />
                            <input
                                type="text"
                                value={theme.accent}
                                onChange={(e) => setTheme((cur) => ({ ...cur, accent: e.target.value }))}
                                className="flex-1 bg-background-dark border border-white/10 rounded px-3 text-white text-sm"
                            />
                        </div>
                    </div>

                    <div className="space-y-4">
                        <div className="flex items-center justify-between text-sm font-medium text-white/80">
                            <span>Theme Presets</span>
                            <button
                                onClick={handleReset}
                                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 text-white/60 hover:bg-white/10 hover:text-white transition-colors"
                            >
                                <RotateCcw className="size-4" />
                                Reset defaults
                            </button>
                        </div>

                        {/* Dark Themes Category */}
                        <div className="rounded-lg border border-white/10 overflow-hidden">
                            <button
                                onClick={() => setDarkExpanded(!darkExpanded)}
                                className="w-full flex items-center justify-between gap-2 px-3 py-2.5 bg-white/5 hover:bg-white/10 transition-colors"
                            >
                                <div className="flex items-center gap-2 text-xs font-semibold text-white/70 uppercase tracking-wider">
                                    <Moon className="size-3.5" />
                                    <span>Dark Themes</span>
                                    <span className="text-white/40 font-normal">({darkPresets.length})</span>
                                </div>
                                <ChevronDown className={`size-4 text-white/50 transition-transform duration-200 ${darkExpanded ? 'rotate-180' : ''}`} />
                            </button>
                            <div className={`transition-all duration-200 ease-in-out ${darkExpanded ? 'max-h-[200px] opacity-100' : 'max-h-0 opacity-0 overflow-hidden'}`}>
                                <div className="p-2 max-h-[180px] overflow-y-auto scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent hover:scrollbar-thumb-white/20">
                                    <div className="grid grid-cols-2 gap-2">
                                        {darkPresets.map(([key, preset]) => (
                                            <button
                                                key={key}
                                                onClick={() => setTheme(preset)}
                                                className="group flex items-center gap-2.5 p-2.5 rounded-lg border border-white/10 bg-background-dark/50 hover:border-primary/60 hover:bg-white/5 transition-all text-left"
                                            >
                                                <div className="flex gap-1 shrink-0">
                                                    <span className="h-5 w-5 rounded border border-white/10 shadow-inner" style={{ backgroundColor: preset.background }} />
                                                    <span className="h-5 w-5 rounded border border-white/10 shadow-inner" style={{ backgroundColor: preset.surface }} />
                                                    <span className="h-5 w-5 rounded border border-white/10 shadow-inner" style={{ backgroundColor: preset.accent }} />
                                                </div>
                                                <div className="flex-1 min-w-0">
                                                    <p className="text-xs font-semibold text-white group-hover:text-primary transition-colors truncate">
                                                        {humanizePresetName(key)}
                                                    </p>
                                                    <p className="text-[10px] text-white/40 truncate">{preset.description}</p>
                                                </div>
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Light Themes Category */}
                        <div className="rounded-lg border border-white/10 overflow-hidden">
                            <button
                                onClick={() => setLightExpanded(!lightExpanded)}
                                className="w-full flex items-center justify-between gap-2 px-3 py-2.5 bg-white/5 hover:bg-white/10 transition-colors"
                            >
                                <div className="flex items-center gap-2 text-xs font-semibold text-white/70 uppercase tracking-wider">
                                    <Sun className="size-3.5" />
                                    <span>Light Themes</span>
                                    <span className="text-white/40 font-normal">({lightPresets.length})</span>
                                </div>
                                <ChevronDown className={`size-4 text-white/50 transition-transform duration-200 ${lightExpanded ? 'rotate-180' : ''}`} />
                            </button>
                            <div className={`transition-all duration-200 ease-in-out ${lightExpanded ? 'max-h-[200px] opacity-100' : 'max-h-0 opacity-0 overflow-hidden'}`}>
                                <div className="p-2 max-h-[160px] overflow-y-auto scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent hover:scrollbar-thumb-white/20">
                                    <div className="grid grid-cols-2 gap-2">
                                        {lightPresets.map(([key, preset]) => (
                                            <button
                                                key={key}
                                                onClick={() => setTheme(preset)}
                                                className="group flex items-center gap-2.5 p-2.5 rounded-lg border border-white/10 bg-background-dark/50 hover:border-primary/60 hover:bg-white/5 transition-all text-left"
                                            >
                                                <div className="flex gap-1 shrink-0">
                                                    <span className="h-5 w-5 rounded border border-black/10 shadow-inner" style={{ backgroundColor: preset.background }} />
                                                    <span className="h-5 w-5 rounded border border-black/10 shadow-inner" style={{ backgroundColor: preset.surface }} />
                                                    <span className="h-5 w-5 rounded border border-black/10 shadow-inner" style={{ backgroundColor: preset.accent }} />
                                                </div>
                                                <div className="flex-1 min-w-0">
                                                    <p className="text-xs font-semibold text-white group-hover:text-primary transition-colors truncate">
                                                        {humanizePresetName(key)}
                                                    </p>
                                                    <p className="text-[10px] text-white/40 truncate">{preset.description}</p>
                                                </div>
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="flex gap-3 mt-8 pt-6 border-t border-white/10">
                    <div className="flex-1"></div>
                    <button
                        onClick={onClose}
                        className="px-4 py-2 rounded-lg bg-white/10 text-white hover:bg-white/20 transition-colors text-sm font-medium"
                    >
                        Done
                    </button>
                </div>
            </div>
        </div>
    );
};
