# SparseChron — Design

UI/UX guidelines for the Viser interactive 3D viewer — the only visual
interface in the project.

---

## 1. Design Direction

| Property | Choice |
|---|---|
| **Theme** | Dark — dark background, light text, matching 3D viewer conventions (Blender, MeshLab, Nerfstudio) |
| **Feel** | Professional research tool — clean, functional, no decorative elements |
| **Priority** | The 3D scene is the star. UI controls exist to serve the scene, not compete with it |

---

## 2. Color Palette

### Primary Colors

| Role | Hex | Usage |
|---|---|---|
| **Background** | `#1a1a2e` | Viewer background, panel backgrounds |
| **Surface** | `#242442` | Sidebar, control panels, card backgrounds |
| **Surface hover** | `#2e2e52` | Hover states on interactive elements |
| **Border** | `#3a3a5c` | Panel borders, dividers |

### Accent Colors

| Role | Hex | Usage |
|---|---|---|
| **Primary accent** | `#6366F1` (Indigo) | Active buttons, selected states, slider fill |
| **Secondary accent** | `#3B82F6` (Blue) | Links, secondary buttons, progress indicators |
| **Primary hover** | `#818CF8` | Hover state on primary accent elements |

### Text Colors

| Role | Hex | Usage |
|---|---|---|
| **Primary text** | `#E2E8F0` | Labels, headings, main content |
| **Secondary text** | `#94A3B8` | Descriptions, stats, metadata |
| **Muted text** | `#64748B` | Disabled states, placeholders |

### Semantic Colors

| Role | Hex | Usage |
|---|---|---|
| **Success** | `#22C55E` | Snapshot saved confirmation |
| **Warning** | `#F59E0B` | VRAM usage warning |
| **Error** | `#EF4444` | Load failure messages |

### 3D Viewport

| Role | Hex | Usage |
|---|---|---|
| **Viewport background** | `#0f0f1a` | Behind the Gaussian splat scene |
| **Grid lines** | `#2a2a44` (20% opacity) | Optional ground plane grid |
| **Axis gizmo** | R: `#EF4444`, G: `#22C55E`, B: `#3B82F6` | XYZ axis indicator |

---

## 3. Typography

| Element | Font | Size | Weight |
|---|---|---|---|
| **Panel headings** | System sans-serif (Viser default) | 14px | 600 (semi-bold) |
| **Labels** | System sans-serif | 12px | 400 (regular) |
| **Stats/numbers** | System monospace | 12px | 400 |
| **Tooltips** | System sans-serif | 11px | 400 |

Viser uses the browser's system font stack — no custom font loading needed.
Monospace for numerical stats ensures alignment.

---

## 4. Viewer Layout

```
┌─────────────────────────────────────────────────────────┐
│  SparseChron Viewer                          [≡] [─] [×]│
├────────────┬────────────────────────────────────────────┤
│            │                                            │
│  CONTROLS  │                                            │
│            │                                            │
│ ┌────────┐ │              3D VIEWPORT                   │
│ │ Time   │ │                                            │
│ │ ══●═══ │ │         (Gaussian Splat Scene)             │
│ │ 0.42   │ │                                            │
│ └────────┘ │                                            │
│            │                                            │
│ [▶ Play ]  │                                            │
│            │                                            │
│ ┌────────┐ │                                            │
│ │ View   │ │                                            │
│ │ □ BG   │ │                                 ┌───┐      │
│ │ □ Grid │ │                                 │xyz│      │
│ └────────┘ │                                 └───┘      │
│            │                                 axis       │
│ ┌────────┐ │                                            │
│ │ Stats  │ │                                            │
│ │ G: 185K│ │                                            │
│ │ S: 133K│ │                                            │
│ │ D:  52K│ │                                            │
│ │ FPS: 30│ │                                            │
│ └────────┘ │                                            │
│            │                                            │
│ [📷 Snap ] │                                            │
│ [⟲ Reset ] │                                            │
│            │                                            │
├────────────┴────────────────────────────────────────────┤
│  Scene: lego  │  Iteration: 30000  │  PSNR: 29.4 dB    │
└─────────────────────────────────────────────────────────┘
```

### Layout Rules

- **Sidebar**: Fixed width ~200px, left side. Contains all controls
- **Viewport**: Fills remaining space. This is where the 3D scene renders
- **Status bar**: Bottom strip showing scene name, training iteration, and
  metrics — informational only, not interactive
- **Axis gizmo**: Small XYZ indicator in bottom-right of viewport

---

## 5. UI Controls Specification

### Time Slider

- Horizontal slider in the sidebar
- Range: `[0.0, 1.0]` (normalized timestep)
- Current value displayed below as a number
- Slider track: `#3a3a5c` (border color)
- Slider fill (left of thumb): `#6366F1` (primary accent)
- Slider thumb: `#E2E8F0` (primary text), circular

### Play / Pause Button

- Toggle button: `▶ Play` / `⏸ Pause`
- Default state: paused
- When playing: auto-increment timestep at ~10 FPS, loop from 1.0 back to 0.0
- Button uses primary accent color when playing (active state)

### Background Toggle

- Checkbox: `☐ Show Background` / `☑ Show Background`
- When unchecked: static Gaussians (`is_dynamic == False`) are hidden
- Useful for isolating dynamic content during demo

### Stats Panel

- Read-only display, updates every frame
- Shows: total Gaussian count, static count, dynamic count, render FPS
- Monospace font for numerical alignment

### Snapshot Button

- Click → renders current view at full resolution → saves as PNG
- Brief green flash / `✓ Saved` confirmation text (2 seconds)
- Saves to `outputs/<experiment>/snapshots/snap_<timestamp>.png`

### Reset Camera Button

- Returns camera to default position (looking at scene center, distance = scene bounding box diagonal)
- Smooth animated transition (0.3s ease-out) if Viser supports it,
  otherwise instant

---

## 6. Interaction Design

### Camera Controls (Viser defaults)

| Input | Action | Note |
|---|---|---|
| Left-click + drag | Orbit | Rotate around scene center |
| Right-click + drag | Pan | Translate camera laterally |
| Scroll wheel | Zoom | Dolly in/out |
| Double-click | Focus | Center on clicked point |

### Responsiveness

- Viewer must maintain **≥ 15 FPS** at 200K Gaussians for usable interaction
- If FPS drops below 15, display a warning in the stats panel
- On CPU (local Windows), reduce render resolution by 2× to maintain
  interactivity

---

## 7. States & Feedback

| State | Visual Feedback |
|---|---|
| **Loading model** | Spinner in viewport center + "Loading model..." text |
| **Model loaded** | Scene appears, stats populate, controls become interactive |
| **Playing animation** | Play button shows active (accent color), timestep auto-advances |
| **Snapshot saved** | Brief `✓ Saved` text near the snapshot button (green, fades after 2s) |
| **Error loading** | Red error message in viewport center with file path |
| **Low FPS warning** | Amber text in stats panel: `⚠ Low FPS` |

---

*Derived from user style preferences (dark theme, blue-purple accents) and the
viewer requirements in
[PRD.md](file:///c:/Users/odeda/Desktop/Projects/IVP%20Project/PRD.md) §F6 and
[AppFlow.md](file:///c:/Users/odeda/Desktop/Projects/IVP%20Project/AppFlow.md) Flow 4.*
