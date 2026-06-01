# Ví dụ: LaTeX → Plain text cho Markdown

## Biến / ký hiệu

| ❌ LaTeX | ✅ Plain text |
|---|---|
| `\(x_{source}\)` | ảnh nguồn `x_source` hoặc **x_source** |
| `\(y_{edit}\)` | edit prompt `y_edit` |
| `\(F_\theta\)` | mạng inversion `F_theta` |
| `\(\hat{\epsilon}\)` | inverted noise `eps_hat` hoặc ε̂ (Unicode) |
| `\(s_y\)` | hệ số `s_y` |
| `\(s_{non\text{-}edit}\)` | hệ số `s_non-edit` |
| `\(M\)` | editing mask **M** |

## Công thức

| ❌ LaTeX block | ✅ Plain text |
|---|---|
| `\[\hat{\epsilon} = F_\theta(z, c_y)\]` | `eps_hat = F_theta(z, c_y)` |
| `\(M = \text{normalize}(\|\hat{\epsilon}_{source} - \hat{\epsilon}_{edit}\|)\)` | `M = normalize(abs(eps_hat_source - eps_hat_edit))` |

## Đoạn mô tả

**❌ LaTeX:**

> SwiftEdit học mạng \(F_\theta\) ánh xạ ảnh nguồn về latent noise \(\hat{\epsilon}\), điều khiển qua \(s_y\), \(s_{edit}\), \(s_{non\text{-}edit}\).

**✅ Plain text:**

> SwiftEdit học mạng inversion `F_theta` ánh xạ ảnh nguồn về latent noise `eps_hat`, điều khiển cường độ qua các hệ số `s_y`, `s_edit`, `s_non-edit`.

## Bảng input/output

**❌ LaTeX:**

| Input | Mô tả |
|---|---|
| **Source image** \(x_{source}\) | Ảnh gốc |

**✅ Plain text:**

| Input | Mô tả |
|---|---|
| **Source image** (`x_source`) | Ảnh gốc RGB cần chỉnh sửa |

## Pipeline (ASCII — luôn OK)

```
Ảnh nguồn + source/edit prompt
    -> One-step Inversion (F_theta) -> eps_hat
    -> Self-guided mask M
    -> G_IP + ARaM -> ảnh đã chỉnh sửa
```

## Ký tự Unicode hữu ích

| Ý nghĩa | Ký tự |
|---|---|
| epsilon | ε |
| theta | θ |
| sigma | σ |
| nhân / times | × |
| mũ tên | → |
| xấp xỉ | ≈ |
| lớn hơn bằng | ≥ |

Tránh overuse Unicode — ưu tiên `backtick` cho tên biến code.
