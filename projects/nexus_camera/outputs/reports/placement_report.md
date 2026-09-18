# 3D Element Placement & Viewpoint Specification
**Source Target:** `C:\Users\PC\myapps\Blender works\projects\nexus_camera\reference\front_view.png` | **Pipeline Version:** `4.0.0`

## 1. Global Coordinate System & Bounds
- **Coordinate Origin:** BASE_CENTER_ON_GROUND (Z=0.0)
- **Axes:** X=RIGHT, Y=FORWARD (camera facing), Z=VERTICAL UP
- **Dimensions:** 2.0 x 2.0 x 1.502 BU (Aspect Ratio: 0.75)

## 2. Component Layout & Z-Elevations
| Component | Category | Z Range (Norm) | Height (BU) | Center (X, Z) BU | Visual Description |
|:---|:---|:---|:---|:---|:---|
| **Upper Structure & Viewfinder Prism** | `closure` | `[0.841 - 1.000]` | `0.24` | `(0.00, 1.38)` |  |
| **Camera Chassis** | `closure` | `[0.132 - 0.841]` | `1.06` | `(0.00, 0.73)` |  |
| **Handgrip Rubber** | `enclosure` | `[0.099 - 0.132]` | `0.05` | `(0.00, 0.17)` |  |
| **Lens Barrel & Focus Ring** | `enclosure` | `[0.066 - 0.099]` | `0.05` | `(0.00, 0.12)` |  |
| **Front Optical Glass Element** | `enclosure` | `[0.033 - 0.066]` | `0.05` | `(0.00, 0.07)` |  |
| **Base Section** | `structural_base` | `[0.000 - 0.033]` | `0.05` | `(0.00, 0.03)` |  |

## 3. Calibrated Blender 3D Transform Table
| Object Name | Location [X, Y, Z] | Scale / Half-Extents | Dimensions [W, D, H] |
|:---|:---|:---|:---|
| `Upper Structure & Viewfinder Prism` | `[0.00, 0.00, 1.38]` | `[1.00, 1.00, 0.12]` | `[2.00, 2.00, 0.24]` |
| `Camera Chassis` | `[0.00, 0.00, 0.73]` | `[1.00, 1.00, 0.53]` | `[2.00, 2.00, 1.06]` |
| `Handgrip Rubber` | `[0.00, 0.00, 0.17]` | `[1.00, 1.00, 0.03]` | `[2.00, 2.00, 0.05]` |
| `Lens Barrel & Focus Ring` | `[0.00, 0.00, 0.12]` | `[1.00, 1.00, 0.03]` | `[2.00, 2.00, 0.05]` |
| `Front Optical Glass Element` | `[0.00, 0.00, 0.07]` | `[1.00, 1.00, 0.03]` | `[2.00, 2.00, 0.05]` |
| `Base Section` | `[0.00, 0.00, 0.03]` | `[1.00, 1.00, 0.03]` | `[2.00, 2.00, 0.05]` |
