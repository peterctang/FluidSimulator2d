#include<math.h>
#include<vector>
#include<iostream>
#include<omp.h>
#include "CImg.h"
#include <sstream>
#include <string>
#include <iomanip>  // Include <iomanip> for std::setfill and std::setw
#include <cstdlib>

using namespace std;

#define IX(i, j) ((i) + (N + 2) * (j))

void add_source(int N, float* x, float* s, float dt);
void diffuse_bad(int N, int b, float* x, float* x0, float diff, float dt);
void diffuse(int N, int b, float* x, float* x0, float diff, float dt);
void advect(int N, int b, float* d, float* d0, float* u, float* v, float dt);
void dens_step(int N, float* x, float* x0, float* u, float* v, float diff, float dt);
void vel_step(int N, float* u, float* v, float* u0, float* v0, float visc, float dt);
void project(int N, float* u, float* v, float* p, float* div);
void set_bnd(int N, int b, float* x);

int ScalarIndex(int i_, int j_);
int VxIndex(int i_, int j_);
int VyIndex(int i_, int j_);

float Distance(float x_, float y_);
float Cos(float x_, float y_);
float Sin(float x_, float y_);

void add_source(int N, float* x, float* s, float dt)
{
	int i, size = (N + 2) * (N + 2);
	for (i = 0; i < size; i++) x[i] += dt * s[i];
}

void diffuse_bad(int N, int b, float* x, float* x0, float diff, float dt)
{
	int i, j;
	float a = dt * diff * N * N;
	for (i = 1; i <= N; i++) {
		for (j = 1; j <= N; j++) {
			x[IX(i, j)] = x0[IX(i, j)] + a * (x0[IX(i - 1, j)] + x0[IX(i + 1, j)] +
				x0[IX(i, j - 1)] + x0[IX(i, j + 1)] - 4 * x0[IX(i, j)]);
		}
	}
	set_bnd(N, b, x);
}

void diffuse(int N, int b, float* x, float* x0, float diff, float dt)
{
	int i, j, k;
	float a = dt * diff * N * N;
	for (k = 0; k < 20; k++) {
		for (i = 1; i <= N; i++) {
			for (j = 1; j <= N; j++) {
				x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
					x[IX(i, j - 1)] + x[IX(i, j + 1)])) / (1 + 4 * a);
			}
		}
		set_bnd(N, b, x);
	}
}

void advect(int N, int b, float* d, float* d0, float* u, float* v, float dt)
{
	int i, j, i0, j0, i1, j1;
	float x, y, s0, t0, s1, t1, dt0;
	dt0 = dt * N;
	for (i = 1; i <= N; i++) {
		for (j = 1; j <= N; j++) {
			x = i - dt0 * u[IX(i, j)]; y = j - dt0 * v[IX(i, j)];
			if (x < 0.5) x = 0.5; if (x > N + 0.5) x = N + 0.5; i0 = (int)x; i1 = i0 + 1;
			if (y < 0.5) y = 0.5; if (y > N + 0.5) y = N + 0.5; j0 = (int)y; j1 = j0 + 1;
			s1 = x - i0; s0 = 1 - s1; t1 = y - j0; t0 = 1 - t1;
			d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
				s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
		}
	}
	set_bnd(N, b, d);
}

#define SWAP(x0,x) {float *tmp=x0;x0=x;x=tmp;}

void dens_step(int N, float* x, float* x0, float* u, float* v, float diff,
	float dt)
{
	add_source(N, x, x0, dt);
	SWAP(x0, x); 
	diffuse(N, 0, x, x0, diff, dt);
	SWAP(x0, x); advect(N, 0, x, x0, u, v, dt); 
}

void vel_step(int N, float* u, float* v, float* u0, float* v0,
	float visc, float dt)
{
	add_source(N, u, u0, dt); add_source(N, v, v0, dt);
	SWAP(u0, u); diffuse(N, 1, u, u0, visc, dt);
	SWAP(v0, v); diffuse(N, 2, v, v0, visc, dt);
	project(N, u, v, u0, v0);
	SWAP(u0, u); SWAP(v0, v);
	advect(N, 1, u, u0, u0, v0, dt); advect(N, 2, v, v0, u0, v0, dt);
	project(N, u, v, u0, v0);
}

void project(int N, float* u, float* v, float* p, float* div)
{
	int i, j, k;
	float h;
	h = 1.0 / N;
	for (i = 1; i <= N; i++) {
		for (j = 1; j <= N; j++) {
			div[IX(i, j)] = -0.5 * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] +
				v[IX(i, j + 1)] - v[IX(i, j - 1)]);
			p[IX(i, j)] = 0;
		}
	}
	set_bnd(N, 0, div); set_bnd(N, 0, p);
	for (k = 0; k < 20; k++) {
		for (i = 1; i <= N; i++) {
			for (j = 1; j <= N; j++) {
				p[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] +
					p[IX(i, j - 1)] + p[IX(i, j + 1)]) / 4;
			}
		}
		set_bnd(N, 0, p);
	}
	for (i = 1; i <= N; i++) {
		for (j = 1; j <= N; j++) {
			u[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
			v[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
		}
	}
	set_bnd(N, 1, u); set_bnd(N, 2, v);
}

void set_bnd(int N, int b, float* x) {
	for (int i = 1; i <= N; i++) {
		x[IX(0, i)] = (b == 1) ? -x[IX(1, i)] : x[IX(1, i)];
		x[IX(N + 1, i)] = (b == 1) ? -x[IX(N, i)] : x[IX(N, i)];
		x[IX(i, 0)] = (b == 2) ? -x[IX(i, 1)] : x[IX(i, 1)];
		x[IX(i, N + 1)] = (b == 2) ? -x[IX(i, N)] : x[IX(i, N)];
	}

	x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
	x[IX(0, N + 1)] = 0.5 * (x[IX(1, N + 1)] + x[IX(0, N)]);
	x[IX(N + 1, 0)] = 0.5 * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
	x[IX(N + 1, N + 1)] = 0.5 * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
}

#define N 1022
#define NON_ZERO_REGION 100
#define NON_BOUND 300

float cell_width = 1.0;
float visual_width = 800.0/ (N + 2);
float dt = 0.0001;
float diff = 0.0000;
float visc = 0.000011;

float density_latest[(N + 2) * (N + 2)];
float density_backup[(N + 2) * (N + 2)];

float pressure_bar[(N + 2) * (N + 2)];
float divV[(N + 2) * (N + 2)];


float vx_latest[(N + 2) * (N + 2)];
float vx_backup[(N + 2) * (N + 2)];

float vy_latest[(N + 2) * (N + 2)];
float vy_backup[(N + 2) * (N + 2)];


float total;

float Distance(float x_, float y_)
{
	return sqrt((x_ - (N + 2) / 2 * cell_width) * (x_ - (N + 2) / 2 * cell_width) + (y_ - (N + 2) / 2 * cell_width) * (y_ - (N + 2) / 2 * cell_width));
}


float Sin(float x_, float y_)
{
	if (Distance(x_, y_) == 0)
	{
		return 0;
	}
	else
	{
		return (y_ - (N + 2) / 2 * cell_width) / Distance(x_, y_);
	}
}

float Cos(float x_, float y_)
{

	if (Distance(x_, y_) == 0)
	{
		return 0.0;
	}
	else
	{
		return (x_ - (N + 2) / 2 * cell_width) / Distance(x_, y_);
	}

}

int ScalarIndex(int i_, int j_)
{
	return j_ * (N + 2) + i_;
}

int VxIndex(int i_, int j_)
{
	return j_ * (N + 2) + i_;
}

int VyIndex(int i_, int j_)
{
	return j_ * (N + 2) + i_;
}

void SyncDensity()
{

	for (size_t i = 0; i < (N + 2) * (N + 2); i++)
	{
		density_backup[i] = density_latest[i];
		//std::cout << "thread_num: " << omp_get_thread_num() << std::endl;
	}
}

void SyncV()
{
	for (size_t i = 0; i < (N + 1) * (N + 2); i++)
	{
		vx_backup[i] = vx_latest[i];
		vy_backup[i] = vy_latest[i];
	}

}

void Draw(int pixel, int frame) {

	cimg_library::CImg<unsigned char> image(pixel, pixel, 1, 3, 0);  // Create a black image

	float x, y, width_of_cell = pixel / (N + 2);
	float grey;
	//painter.setPen(QPen(QColor(0, 0, 0, 0), 1));
	for (size_t i = 0; i < N + 2; i++)
	{
		x = 0 + width_of_cell * i;
		for (size_t j = 0; j < N + 2; j++)
		{
			y = 0 + width_of_cell * j;

			grey = 1*255 * density_latest[ScalarIndex(i, j)], 0, 255;

			if (grey < 0) {
				grey = 0; // If x is less than the minimum, set it to the minimum.
			}
			else if (grey > 255) {
				grey = 255; // If x is greater than the maximum, set it to the maximum.
			}

			unsigned char color[] = { grey, grey, grey };  // Blue color

			if (x >= 0 && x < image.width() && y >= 0 && y < image.height()) {
				image.draw_point(x, y, color);
	
			}
			
		}

	}

	// Save the image to your local PC
	//const char* filename = "pixel_drawing.bmp";
	//image.save(filename);

	std::ostringstream filename;
	filename << "frame_" << std::setfill('0') << std::setw(4) << frame << ".bmp"; // e.g., frame_0000.png
	image.save(filename.str().c_str());
}

int main()
{

	{
		for (size_t i = 0; i < N + 2; i++)
		{

			for (size_t j = 0; j < N + 2; j++)
			{

				if (Distance(i * cell_width, j * cell_width) < NON_ZERO_REGION * cell_width)
				{
					density_latest[ScalarIndex(i, j)] = 1.0;
					//density_backup[ScalarIndex(i, j)] = 0.0;
				}
				else
				{
					density_latest[ScalarIndex(i, j)] = 0.0;
					//density_backup[ScalarIndex(i, j)] = 0.0;
				}

				//divV[ScalarIndex(i, j)] = 0.0;
				//pressure_bar[ScalarIndex(i, j)] = 0.0;
			}
		}

		for (size_t i = 0; i < N + 2; i++)
		{
			for (size_t j = 0; j < N + 2; j++)
			{

				if (Distance((j + 0.0) * cell_width, i * cell_width) < NON_ZERO_REGION * cell_width *1.0)
				{
					vx_latest[VxIndex(i, j)] = -Distance(j * cell_width, (i + 0.0) * cell_width) *Sin((i + 0.0) * cell_width, j * cell_width) * 100;
					//vx_latest[VxIndex(j, i)] = 10.0;
				}
				else if (NON_ZERO_REGION * cell_width <= Distance(j  * cell_width, i * cell_width) < NON_BOUND * cell_width * 1.0)
				{
					//vx_latest[VxIndex(i, j)] = Distance(j * cell_width, i * cell_width) * Sin(i * cell_width, j * cell_width) * 0.01;
					vx_latest[VxIndex(i, j)] = -0.05;
				}
				else
				{
					vx_latest[VxIndex(j, i)] = 0.0;
				}


				if (Distance(j * cell_width, (i + 0.0) * cell_width) < NON_ZERO_REGION * cell_width *1.0)
				{
					vy_latest[VyIndex(j, i)] = Distance(j * cell_width, (i + 0.0) * cell_width) *Cos(j * cell_width, (i + 0.0) * cell_width) * 100;
					//vy_latest[VyIndex(j, i)] = 10.0;

				}
				else if (NON_ZERO_REGION * cell_width <= Distance(j * cell_width, i * cell_width) < NON_BOUND * cell_width * 1.0)
				{
					//vy_latest[VyIndex(j, i)] = -Distance(j * cell_width, i * cell_width) * Cos(j * cell_width, i * cell_width) * 0.01;
					vy_latest[VyIndex(j, i)] = -0.05;
				}
				else
				{
					vy_latest[VyIndex(j, i)] = 0.0;
				}




			}

		}

	}

	for (int frame = 0; frame < 240; frame++) {
		//SyncDensity();
		//SyncV();
		//add_source(N, density_latest, density_backup, dt);
		vel_step(N, vx_latest, vy_latest, vx_backup, vy_backup, visc, dt);
		dens_step(N, density_latest, density_backup, vx_latest, vy_latest, diff, dt);
		Draw(N+2, frame);
		
		}
	//Draw(1024, 0);
	//printf("i = % d", i);
//	const char* folderPath = "C:\\Users\\tangc\\source\\repos\\FluidSimulator\\Project1";

	// Output video file name
//	const char* outputVideo = "output_video.mp4";

	// Command to create a video from image frames using FFmpeg
//	const char* ffmpegCommand = "ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p ";
//	std::string cmd = std::string(ffmpegCommand) + folderPath + outputVideo;

	// Execute the FFmpeg command
//	int result = std::system(cmd.c_str());

//	if (result == 0) {
//		std::cout << "Video created successfully." << std::endl;
//	}
//	else {
//		std::cerr << "Error creating the video." << std::endl;
//	}
	return 0;
}