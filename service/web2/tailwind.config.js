export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        "primary": "#00E5FF",
        "background-light": "#f6f6f8",
        "background-dark": "#1A1A1D",
      },
      fontFamily: {
        "display": ["Space Grotesk", "sans-serif"]
      },
    },
  },
  plugins: [],
}

