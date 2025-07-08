/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        'mlbb-gold': '#FFD700',
        'mlbb-blue': '#1E3A8A',
        'mlbb-red': '#DC2626',
        'mlbb-purple': '#7C3AED',
        'mlbb-dark': '#1F2937',
      }
    },
  },
  plugins: [],
}