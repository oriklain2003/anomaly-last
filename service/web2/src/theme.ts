export type ThemeConfig = {
  /** Primary / background color */
  background: string;
  /** Secondary / card & item color */
  surface: string;
  /** Accent / tabs & highlights */
  accent: string;
};

export const THEME_STORAGE_KEY = 'app-theme-colors';

export const DEFAULT_THEME: ThemeConfig = {
  background: '#09090b',
  surface: '#18181b',
  accent: '#00E5FF',
};

export type ThemePreset = ThemeConfig & {
  category: 'dark' | 'light';
  description?: string;
};

export const PRESET_THEMES: Record<string, ThemePreset> = {
  // ═══════════════════════════════════════════════════════════════
  // DARK THEMES
  // ═══════════════════════════════════════════════════════════════
  onyxMidnight: {
    background: '#0E1320',
    surface: '#0A0F1A',
    accent: '#5C8DFF',
    category: 'dark',
    description: 'Deep blue midnight',
  },
  nebulaPulse: {
    background: '#120F1F',
    surface: '#0B0A13',
    accent: '#6AE6DA',
    category: 'dark',
    description: 'Cosmic teal energy',
  },
  stealthGraphite: {
    background: '#1A1F2B',
    surface: '#131720',
    accent: '#FFB757',
    category: 'dark',
    description: 'Tactical warm accent',
  },
  carbonRedline: {
    background: '#0D0D0D',
    surface: '#161616',
    accent: '#FF3B30',
    category: 'dark',
    description: 'High alert mode',
  },
  auroraDrift: {
    background: '#0C1D24',
    surface: '#081418',
    accent: '#4CE6A5',
    category: 'dark',
    description: 'Northern lights',
  },
  lunarMirage: {
    background: '#0C0F14',
    surface: '#1B2633',
    accent: '#4DD2FF',
    category: 'dark',
    description: 'Futuristic moon-base',
  },
  royalNebula: {
    background: '#11081A',
    surface: '#26143A',
    accent: '#FF4FA7',
    category: 'dark',
    description: 'Cosmic purple elegance',
  },
  desertFalcon: {
    background: '#1A1712',
    surface: '#2C241C',
    accent: '#E7B66A',
    category: 'dark',
    description: 'Desert tactical warmth',
  },
  sandstormEmber: {
    background: '#1C1A14',
    surface: '#2A251C',
    accent: '#F28F3B',
    category: 'dark',
    description: 'Glowing ember night',
  },
  obsidianFog: {
    background: '#0A0C0F',
    surface: '#1A1F26',
    accent: '#8AB4F8',
    category: 'dark',
    description: 'Stealthy cyber-military',
  },
  magmaStrike: {
    background: '#120E0E',
    surface: '#1F1717',
    accent: '#FF3B30',
    category: 'dark',
    description: 'Volcanic high alert',
  },
  sinaiNightfall: {
    background: '#0E1118',
    surface: '#19202C',
    accent: '#1BA6CF',
    category: 'dark',
    description: 'Middle-East night sky',
  },
  quantumDrift: {
    background: '#0B0D12',
    surface: '#182029',
    accent: '#7AF0B5',
    category: 'dark',
    description: 'Sci-fi space station',
  },
  dawnPatrol: {
    background: '#191614',
    surface: '#2A2420',
    accent: '#F6C667',
    category: 'dark',
    description: 'Early morning aviation',
  },

  // ═══════════════════════════════════════════════════════════════
  // LIGHT THEMES
  // ═══════════════════════════════════════════════════════════════
  skyboundLight: {
    background: '#F4F6F9',
    surface: '#FFFFFF',
    accent: '#2E8BFF',
    category: 'light',
    description: 'Clean aviation dashboard',
  },
  citrusNova: {
    background: '#F5F3E8',
    surface: '#FFFFFF',
    accent: '#FFC93C',
    category: 'light',
    description: 'Energetic premium bright',
  },
  sakuraFrost: {
    background: '#F5F7FA',
    surface: '#FFFFFF',
    accent: '#FF5E99',
    category: 'light',
    description: 'Elegant soft pink',
  },
  tropicWave: {
    background: '#F2FAF9',
    surface: '#FFFFFF',
    accent: '#00C4A6',
    category: 'light',
    description: 'Summer turquoise energy',
  },
  arcticVision: {
    background: '#F6F9FC',
    surface: '#FFFFFF',
    accent: '#3DA5F4',
    category: 'light',
    description: 'Scandinavian icy clarity',
  },
};

const hexToRgb = (hex: string) => {
  const shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
  const normalized = hex.replace(shorthandRegex, (_m, r, g, b) => r + r + g + g + b + b);
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(normalized);
  return result ? `${parseInt(result[1], 16)} ${parseInt(result[2], 16)} ${parseInt(result[3], 16)}` : null;
};

/** 
 * Calculate relative luminance of a hex color
 * Returns value between 0 (black) and 1 (white)
 */
const getLuminance = (hex: string): number => {
  const rgb = hexToRgb(hex);
  if (!rgb) return 0;
  const [r, g, b] = rgb.split(' ').map(Number);
  // Relative luminance formula (ITU-R BT.709)
  const [rs, gs, bs] = [r, g, b].map((c) => {
    const sRGB = c / 255;
    return sRGB <= 0.03928 ? sRGB / 12.92 : Math.pow((sRGB + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
};

/** Check if a color is considered "light" (luminance > 0.5) */
export const isLightColor = (hex: string): boolean => getLuminance(hex) > 0.5;

export const applyTheme = (theme: ThemeConfig) => {
  if (typeof document === 'undefined') return;
  const root = document.documentElement;

  const backgroundRgb = hexToRgb(theme.background);
  const surfaceRgb = hexToRgb(theme.surface);
  const accentRgb = hexToRgb(theme.accent);

  if (backgroundRgb) root.style.setProperty('--color-background', backgroundRgb);
  if (surfaceRgb) root.style.setProperty('--color-surface', surfaceRgb);
  if (accentRgb) root.style.setProperty('--color-primary', accentRgb);

  // Detect if theme is light or dark and set text colors accordingly
  const isLight = isLightColor(theme.background);
  root.setAttribute('data-theme', isLight ? 'light' : 'dark');
  
  // Set text color variables based on theme
  if (isLight) {
    root.style.setProperty('--color-text', '15 23 42');           // slate-900
    root.style.setProperty('--color-text-muted', '71 85 105');    // slate-500
    root.style.setProperty('--color-border', '203 213 225');      // slate-300
  } else {
    root.style.setProperty('--color-text', '255 255 255');        // white
    root.style.setProperty('--color-text-muted', '148 163 184');  // slate-400
    root.style.setProperty('--color-border', '51 65 85');         // slate-700
  }

  try {
    localStorage.setItem(THEME_STORAGE_KEY, JSON.stringify(theme));
  } catch (error) {
    console.warn('Unable to persist theme colors', error);
  }
};

export const loadSavedTheme = (): ThemeConfig => {
  if (typeof localStorage === 'undefined') return DEFAULT_THEME;
  const savedConfig = localStorage.getItem(THEME_STORAGE_KEY);
  if (!savedConfig) return DEFAULT_THEME;

  try {
    const parsed = JSON.parse(savedConfig);
    return {
      background: parsed.background || DEFAULT_THEME.background,
      surface: parsed.surface || DEFAULT_THEME.surface,
      accent: parsed.accent || parsed.primary || DEFAULT_THEME.accent,
    };
  } catch (error) {
    console.error('Failed to parse saved theme settings', error);
    return DEFAULT_THEME;
  }
};

export const ensureThemeInitialized = () => {
  const theme = loadSavedTheme();
  applyTheme(theme);
};

