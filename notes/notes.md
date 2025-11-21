Gaussian parameters:
- Position: (x, y, z)
- Scale: (sx, sy, sz)
- Rotation: quaternion (q0, q1, q2, q3), normalized
- Opacity: alpha ∈ (0,1)
- Color: RGB or SH coeffs

Requirements:

1. Change the cost function, create your own Cost function, not even change. Create your own cost function and show me that you can change the cost function to achieve some purpose. It can be some construction from some paper. I don't care which paper. But you have to have implemented a lot and you need to be able to explain to me the cost function properly. 

2. And then the other thing which is easier is you take two Gaussians, right, and then just do the spectral norm of that, which is basically just running the eigenvector decomposition, or eigenvalue the decomposition on that, and then showing me the eigenvalues and showing me clear differences.

Instead of traditional polygons(mesh) or neural networks like NeRF, Gaussian Splatting uses ellipsoids named as gaussian splats

Each 3D point cloud is converted into Gaussian Splat, with parameters as 

| **Parameter** | **Description** |
| --- | --- |
| **Position (mu)** | The x, y, z coordinates where the splat is centered in 3D space. |
| **Covariance (Sigma)** | A matrix that defines the **shape, size, and orientation** of the ellipsoid (how it's stretched or rotated). |
| **Color** | The color of the splat, often represented using **Spherical Harmonics (SH)** to model how the color changes depending on the viewing direction (view-dependent appearance). |
| **Opacity (alpha)** | The transparency of the splat, crucial for blending multiple overlapping splats. |

Then, Project the 3D Gaussians onto a 2D image plane (splatting & rasterization). 

Loss functions compares original input images to rendered 2D image

Then using Stochastic Gradient Descent or another optimization algorithm, parameters arae adjusted to minimize the loss function

Covariance eigenvalues - s1 s2 s3 

We basically have Spectral entropy as 

![image.png](image.png)

- When s1=s2=s3, **spectral entropy is maximized** and condition number is minimized.
- Needle-like artifacts correspond to **low spectral entropy and high condition number**.

So we will penalize gaussians with low entropy

Multiply this spectral loss with spectral lambda and add it to our loss from before