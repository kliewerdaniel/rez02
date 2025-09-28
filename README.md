# Daniel Kliewer Portfolio

A modern, production-ready portfolio website built with Next.js 15, TypeScript, and Tailwind CSS. This site serves as a professional resume, project showcase, blog, and art gallery for Daniel Kliewer, an AI Developer and Full-Stack Technologist.

## 🚀 Features

- **Modern Tech Stack**: Next.js 15 with App Router, TypeScript, Tailwind CSS
- **Responsive Design**: Mobile-first design that works on all devices
- **Dark Mode**: Built-in dark/light theme toggle
- **Project Showcase**: GitHub-integrated project display with fallback data
- **Blog System**: MDX-based blog with syntax highlighting and metadata
- **Art Gallery**: Image gallery with lightbox modal functionality
- **Contact Form**: Functional contact form with validation
- **SEO Optimized**: Meta tags, Open Graph, and Twitter Card support
- **Performance**: Optimized images, lazy loading, and caching strategies
- **Accessibility**: ARIA attributes and keyboard navigation support

## 📁 Project Structure

```
daniel-kliewer-portfolio/
├── src/
│   └── app/
│       ├── about/           # About/Resume page
│       ├── art/            # Art gallery page
│       ├── blog/           # Blog system
│       │   ├── blog-[slug]/ # Individual blog posts
│       │   └── page.tsx    # Blog index
│       ├── contact/        # Contact form page
│       ├── projects/       # Projects showcase
│       ├── globals.css     # Global styles
│       ├── layout.tsx      # Root layout with navigation
│       └── page.tsx        # Homepage
├── components/             # Reusable React components
├── content/blog/          # Markdown blog posts
├── data/                  # Cached GitHub data and sources
├── public/               # Static assets
├── netlify/              # Netlify functions
└── scripts/              # Utility scripts
```

## 🛠️ Tech Stack

### Core Technologies
- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Content**: MDX for blog posts
- **Icons**: Heroicons
- **Fonts**: Geist Sans & Geist Mono

### Dependencies
- **UI Components**: Custom components with Tailwind
- **Markdown Processing**: gray-matter, remark
- **GitHub Integration**: octokit
- **State Management**: React useState hooks
- **Form Handling**: React Hook Form principles
- **Image Optimization**: Next.js Image component

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- npm or yarn
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kliewerdaniel/daniel-kliewer-portfolio.git
   cd daniel-kliewer-portfolio
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables** (optional)
   ```bash
   cp .env.example .env.local
   ```

   Add your environment variables:
   ```env
   GITHUB_TOKEN=your_github_token_here
   GOOGLE_ANALYTICS_ID=your_ga_id_here
   ```

4. **Run development server**
   ```bash
   npm run dev
   ```

5. **Open your browser**
   Navigate to [http://localhost:3000](http://localhost:3000)

## 📝 Content Management

### Adding Blog Posts

1. Create a new markdown file in `content/blog/`
   ```markdown
   ---
   title: Your Blog Post Title
   date: '2025-01-01'
   description: Brief description of your post
   tags: ['tag1', 'tag2', 'tag3']
   ---

   # Your Blog Post Content

   Write your content here...
   ```

2. The post will automatically appear in the blog index and be accessible at `/blog/your-post-slug`

### Adding Art Pieces

1. Add images to `public/art/` directory
2. The art gallery will automatically detect and display new images
3. Supported formats: PNG, JPG, JPEG, GIF, WebP

### Updating Resume Content

Edit the experience and skills data in `src/app/about/page.tsx` to update your resume information.

## 🔧 Development

### Available Scripts

```bash
# Development server
npm run dev

# Production build
npm run build

# Start production server
npm run start

# Lint code
npm run lint

# Generate resume PDF (requires Puppeteer)
npm run generate-resume-pdf
```

### Code Quality

- **ESLint**: Configured for Next.js and TypeScript
- **TypeScript**: Strict mode enabled
- **Prettier**: Code formatting

## 🌐 Deployment

### Netlify Deployment

1. **Connect to GitHub**
   - Push your code to GitHub
   - Connect your repository to Netlify

2. **Build Settings**
   - Build command: `npm run build`
   - Publish directory: `dist`
   - Node version: 18

3. **Environment Variables**
   Add these in your Netlify dashboard:
   ```env
   GITHUB_TOKEN=your_github_token
   GOOGLE_ANALYTICS_ID=your_ga_id
   ```

### Manual Deployment

```bash
# Build the project
npm run build

# Deploy to Netlify
npx netlify-cli deploy --prod --dir=dist
```

## 🔐 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GITHUB_TOKEN` | GitHub Personal Access Token for API access | Optional |
| `GOOGLE_ANALYTICS_ID` | Google Analytics tracking ID | Optional |
| `PLAUSIBLE_DOMAIN` | Plausible analytics domain | Optional |

### GitHub Token Setup

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate a new token with `repo` permissions
3. Add it to your Netlify environment variables

## 🎨 Customization

### Styling
- Modify `src/app/globals.css` for global styles
- Update Tailwind configuration in `tailwind.config.ts`
- Component styles are in their respective files

### Content
- **Bio**: Edit `src/app/about/page.tsx`
- **Skills**: Update the skills array in the about page
- **Projects**: Modify fallback data in `src/app/projects/page.tsx`
- **Colors**: Update Tailwind color classes throughout components

### SEO
- Update metadata in `src/app/layout.tsx`
- Modify Open Graph and Twitter Card information
- Add structured data as needed

## 🔍 Features in Detail

### GitHub Integration
- Fetches real repository data from GitHub API
- Falls back to static data if API is unavailable
- Displays project languages, stars, and descriptions
- Links to live demos and source code

### Blog System
- Markdown/MDX support with frontmatter
- Syntax highlighting for code blocks
- SEO-optimized post pages
- RSS feed generation ready
- Tag-based organization

### Art Gallery
- Responsive image grid
- Lightbox modal for full-size viewing
- Automatic image detection
- Metadata support for titles, dates, and descriptions

### Contact Form
- Form validation and error handling
- Success/error state management
- Ready for backend integration
- Accessibility features

## 🚨 Troubleshooting

### Common Issues

1. **Build Errors**
   ```bash
   # Clear cache and reinstall
   rm -rf .next node_modules
   npm install
   npm run build
   ```

2. **GitHub API Rate Limiting**
   - Add GITHUB_TOKEN environment variable
   - Token reduces rate limit from 60 to 5000 requests/hour

3. **Image Loading Issues**
   - Check file paths in `public/` directory
   - Verify image formats are supported
   - Clear Next.js cache: `rm -rf .next`

4. **Blog Posts Not Appearing**
   - Ensure markdown files are in `content/blog/`
   - Check frontmatter format
   - Verify file extensions (.md)

### Performance Optimization

- Images are automatically optimized by Next.js
- Static generation for all pages
- CSS and JS minification
- Font optimization with next/font

## 🤝 Contributing

This is a personal portfolio project, but suggestions and improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Next.js**: React framework
- **Tailwind CSS**: Utility-first CSS framework
- **Heroicons**: Beautiful hand-crafted SVG icons
- **Geist**: Modern font family

---

**Built with ❤️ by Daniel Kliewer**

For questions or collaboration opportunities, please reach out through the contact form or connect via GitHub.
