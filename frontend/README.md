# Qualified Health Frontend

Modern React frontend for the Qualified Health clinical decision support system.

## Features

- ⚡ **Vite** for fast development and builds
- ⚛️ **React 19** with TypeScript
- 🎨 **Tailwind CSS** for styling
- 🎭 **Framer Motion** for animations
- 🗄️ **Zustand** for state management
- 🛣️ **React Router** for navigation

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

The app runs at `http://localhost:3000` and proxies API requests to `http://localhost:8000`.

## Project Structure

```
frontend/
├── public/
│   └── favicon.svg
├── src/
│   ├── components/       # Reusable UI components
│   │   ├── ChatSidebar.tsx
│   │   └── ChatWindow.tsx
│   ├── pages/            # Page components
│   │   ├── LandingPage.tsx
│   │   └── ChatPage.tsx
│   ├── stores/           # Zustand state stores
│   │   └── chatStore.ts
│   ├── lib/              # Utilities
│   │   ├── api.ts        # API client
│   │   └── utils.ts      # Helper functions
│   ├── types/            # TypeScript types
│   │   └── index.ts
│   ├── App.tsx           # Root component
│   ├── main.tsx          # Entry point
│   └── index.css         # Global styles
├── index.html
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── vite.config.ts
```

## Pages

### Landing Page (`/`)

Beautiful landing page with:
- Hero section explaining the product
- Feature highlights
- Three response modes explanation
- NICE guidelines integration info
- Call-to-action to start chatting

### Chat Page (`/chat`)

Full-featured chat interface with:
- Conversation sidebar
- Message history with persistence
- Response type indicators
- Citation display
- Example prompts for new conversations

## State Management

Chat state is managed with Zustand and persisted to localStorage:

```typescript
const { conversations, sendMessage, createConversation } = useChatStore();
```

## API Integration

The API client handles all backend communication:

```typescript
import { sendChatMessage } from '@/lib/api';

const response = await sendChatMessage({
  message: "What are the referral criteria?",
  conversation_id: "...",
});
```

## Styling

Uses Tailwind CSS with a custom healthcare-focused color palette:

- `primary` - Sky blue for main actions
- `accent` - Green for success states
- `trust` - Purple for special elements
- `surface` - Slate for backgrounds

Custom components:
- `.card` - Card container
- `.btn-primary` - Primary button
- `.glass` - Glassmorphism effect
- `.mesh-bg` - Gradient mesh background

## Development

```bash
# Type checking
npx tsc --noEmit

# Linting
npm run lint

# Format check
npx prettier --check src/
```

## Build

```bash
# Production build
npm run build

# Output in dist/
```

The build is optimized with:
- Code splitting
- Tree shaking
- Asset optimization
- Gzip compression ready
