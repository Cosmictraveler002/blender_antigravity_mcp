# 3D Element Placement & Viewpoint Specification
**Source Target:** `C:\Users\PC\myapps\Blender works\projects\nexus_camera\reference\front_view.png` | **Pipeline Version:** `4.0.0`

## 1. Global Coordinate System & Bounds
- **Coordinate Origin:** BASE_CENTER_ON_GROUND (Z=0.0)
- **Axes:** X=RIGHT, Y=FORWARD (camera facing), Z=VERTICAL UP
- **Dimensions:** 2.0 x 2.0 x 1.502 BU (Aspect Ratio: 0.75)

## 2. Component Layout & Z-Elevations
| Component | Category | Z Range (Norm) | Height (BU) | Center (X, Z) BU | Visual Description |
|:---|:---|:---|:---|:---|:---|
| **Upper Structure** | `collar` | `[0.841 - 1.000]` | `0.24` | `(0.00, 1.38)` |  |
| **Main Body** | `substrate` | `[0.086 - 0.841]` | `1.13` | `(0.00, 0.69)` |  |
| **Base Section** | `structural_base` | `[0.000 - 0.086]` | `0.13` | `(0.00, 0.07)` |  |

## 3. Calibrated Blender 3D Transform Table
| Object Name | Location [X, Y, Z] | Scale / Half-Extents | Dimensions [W, D, H] |
|:---|:---|:---|:---|
| `Upper Structure` | `[0.00, 0.00, 1.38]` | `[1.00, 1.00, 0.12]` | `[2.00, 2.00, 0.24]` |
| `Main Body` | `[0.00, 0.00, 0.69]` | `[1.00, 1.00, 0.57]` | `[2.00, 2.00, 1.13]` |
| `Base Section` | `[0.00, 0.00, 0.07]` | `[1.00, 1.00, 0.07]` | `[2.00, 2.00, 0.13]` |
