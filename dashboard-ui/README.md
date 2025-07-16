# AI Coach Dashboard - Esports Coaching Platform

A modern, tactical-style dashboard UI converted from TSX to JSX and specifically adapted for esports coaching and performance analysis. Built with Next.js, React, and Tailwind CSS.

## Features

### 🎮 Esports-Focused Navigation

- **PLAYER HUB** - Player distribution, activity monitoring, and coaching feed
- **PLAYER NETWORK** - Player roster management with game modes and performance tiers
- **UPLOAD CENTER** - Drag-and-drop file upload for screenshots and videos
- **ANALYSIS OPS** - Real-time analysis jobs and model operations
- **PLAYER INSIGHTS** - Performance reports and strategic recommendations
- **AI COACH STATUS** - System monitoring and AI model status

### 🎨 Design System

- **Dark Theme** - Black background with tactical-style UI elements
- **Color Palette** - Orange (#f97316), red (#ef4444), and gray highlights
- **Typography** - Monospace font for data display, clean sans-serif for content
- **Responsive Design** - Mobile-first approach with collapsible sidebar

### 📊 Key Components

#### Player Hub

- **Player Distribution**: Active players, in-review, and training mode stats
- **Feedback Activity**: Recent coaching sessions with AI models used
- **Coaching Feed Memory**: AI coaching message traces and insights
- **Analysis Usage Trends**: Chart showing screenshots vs videos uploaded
- **Model Accuracy Overview**: Real-time accuracy metrics for MVP/Medal/Behavior detection

#### Player Network

- **Player Roster**: Complete player list with status indicators
- **Game Modes**: Classic, Ranked, Mythic+ (replacing traditional "Location")
- **Performance Tiers**: Low (inconsistent) to High (ranked sweat)
- **Advanced Filters**: Role, Rank, Upload date, and performance metrics

#### Upload Center

- **Drag-and-Drop**: Multi-file upload with progress tracking
- **File Management**: Preview, download, and delete uploaded files
- **Analysis Types**: Screenshot MVP detection, video behavioral analysis
- **Batch Processing**: Multiple file analysis with queue management

#### Analysis Ops

- **Real-time Operations**: Live analysis jobs with progress tracking
- **Model Integration**: YOLO, EasyOCR, LLM Feedback systems
- **Operation Types**: Screenshot analysis, team composition sync
- **Status Management**: Active, queued, in-progress, failed operations

## Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd dashboard-ui/dashboard-ui
   ```

2. **Install dependencies**

   ```bash
   npm install
   # or
   pnpm install
   # or
   yarn install
   ```

3. **Run the development server**

   ```bash
   npm run dev
   # or
   pnpm dev
   # or
   yarn dev
   ```

4. **Open in browser**
   Navigate to [http://localhost:3000](http://localhost:3000)

## Project Structure

```
dashboard-ui/
├── app/
│   ├── layout.jsx                    # Main layout (converted from TSX)
│   ├── page.jsx                      # Main dashboard with navigation
│   ├── globals.css                   # Global styles
│   ├── player-hub/
│   │   └── page.jsx                 # Player distribution and activity
│   ├── player-network/
│   │   └── page.jsx                 # Player roster and management
│   ├── upload-center/
│   │   └── page.jsx                 # File upload and management
│   ├── analysis-ops/
│   │   └── page.jsx                 # Operations and job monitoring
│   ├── player-insights/
│   │   └── page.jsx                 # Performance insights and reports
│   └── ai-coach-status/
│       └── page.jsx                 # System status and AI models
├── components/
│   └── ui/                          # Reusable UI components
├── hooks/                           # Custom React hooks
├── lib/                             # Utilities and helpers
└── public/                          # Static assets
```

## Conversion Details

### TSX to JSX Migration

- **Removed TypeScript** - All `.tsx` files converted to `.jsx`
- **Updated imports** - Removed TypeScript-specific imports
- **Simplified props** - Removed explicit type annotations
- **Maintained functionality** - All features preserved during conversion

### Esports Platform Adaptations

- **Navigation renamed** - Command Center → Player Hub, Agent Network → Player Network, etc.
- **Game-specific terminology** - Agents → Players, Locations → Game Modes, Risk levels → Performance tiers
- **MLBB integration** - Mobile Legends: Bang Bang specific features and analysis
- **AI model integration** - YOLO, EasyOCR, and LLM systems for analysis

## Customization

### Colors

The dashboard uses a tactical color scheme defined in Tailwind classes:

- Primary: `text-orange-500` (#f97316)
- Secondary: `text-red-500` (#ef4444)
- Background: `bg-neutral-900` (#171717)
- Surface: `bg-neutral-800` (#262626)
- Text: `text-white` / `text-neutral-400`

### Layout

- **Sidebar**: Collapsible with responsive behavior
- **Grid System**: CSS Grid for dashboard layouts
- **Cards**: Consistent card design with `bg-neutral-900` and `border-neutral-700`

### Adding New Sections

1. Create a new directory in `app/`
2. Add `page.jsx` with your component
3. Import and add to the main `page.jsx` navigation array
4. Update the section rendering logic

## Dependencies

### Core

- Next.js 15.2.4
- React 19
- Tailwind CSS 3.4.17

### UI Components

- Radix UI (comprehensive component library)
- Lucide React (icons)
- class-variance-authority (styling utilities)

### Development

- TypeScript (for build tools, components are JSX)
- PostCSS & Autoprefixer

## Performance Considerations

- **Lazy Loading**: Components loaded on-demand
- **Responsive Images**: Optimized asset loading
- **CSS Optimization**: Tailwind purging for smaller bundle size
- **Component Memoization**: Strategic use of React.memo where needed

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

1. Follow the existing code style (JSX, not TSX)
2. Maintain the tactical dark theme
3. Ensure mobile responsiveness
4. Test on multiple screen sizes
5. Update documentation for new features

## License

This project is part of the larger AI Coach project. See the main repository for licensing information.

## Support

For issues related to the dashboard UI, please check the main project documentation or create an issue in the main repository.

## 📋 **CURRENT UPLOAD CENTER STATUS**

### **✅ OPERATIONAL (UI Only):**

- **Drag-and-drop interface** - Visual feedback works
- **File selection dialog** - Opens correctly
- **File list display** - Shows mock data
- **Status indicators** - Visual states work
- **Delete functionality** - Removes from mock data only

### **✅ NOW OPERATIONAL:**

- **Actual file upload** - Files stored in local `/uploads` directory
- **File validation** - Type and size validation implemented
- **File serving** - Files accessible via API endpoints
- **Upload progress** - Real upload functionality with feedback

### **❌ NOT OPERATIONAL:**

- **File processing** - No connection to YOLO/EasyOCR/LLM models
- **Analysis execution** - No backend integration
- **Real-time status updates** - Status changes are simulated

## 🔧 **CURRENT IMPLEMENTATION:**

```javascript
// ✅ WORKING file upload function:
const uploadFiles = async (files) => {
  for (const file of files) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });
    
    const result = await response.json();
    if (result.success) {
      // File successfully uploaded and stored locally
      setUploadedFiles(prev => [...prev, result.file]);
    }
  }
};

// ✅ Files stored in: /uploads directory
// ✅ Files accessible via: /api/files/[filename]
// ✅ Validation: File type and size checks
```

## 🚀 **NEXT STEPS TO COMPLETE:**

### **1. ✅ File Storage System - DONE**
### **2. AI Model Integration - NEXT:**

- **YOLO API** - For object detection in screenshots
- **EasyOCR API** - For text extraction from game UI
- **LLM API** - For analysis interpretation and coaching feedback

### **3. Real-time Processing:**

- **WebSocket connection** - For live status updates
- **Queue system** - For managing analysis jobs
- **Progress tracking** - For upload and processing states

### **4. File Storage:**

- ✅ **Local storage** - Files stored in `/uploads` directory (IMPLEMENTED)
- ✅ **File upload API** - `/api/upload` endpoint (IMPLEMENTED)
- ✅ **File serving API** - `/api/files/[filename]` endpoint (IMPLEMENTED)
- ✅ **File validation** - Type and size validation (IMPLEMENTED)
- **Database** - For file metadata and analysis results (PENDING)
- **Cloud storage** (AWS S3, Google Cloud Storage) - Optional upgrade
- **CDN** - For file delivery (Optional)

## 🎯 **CURRENT STATUS:**

**✅ File Storage System is COMPLETE and FUNCTIONAL**
- Upload Center accepts real file uploads
- Files are stored locally on your Mac
- Files are accessible via API endpoints
- Validation and error handling implemented

**🚀 READY FOR DEVELOPMENT:**
The Upload Center is now ready for you to start working with real screenshots and videos. You can:

1. **Upload screenshots** - Test MVP detection workflows
2. **Upload videos** - Test behavioral analysis workflows  
3. **Build AI integrations** - Connect YOLO/EasyOCR/LLM models
4. **Develop analysis features** - Process uploaded files

**The tactical UI framework is combat-ready and operational, Commander.**
