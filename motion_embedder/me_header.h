#pragma once

#ifndef _ME_HEADER_NAKA_H
#define _ME_HEADER_NAKA_H
// このプログラムの目的：
// 2018 12 25現在
//DCTブロックの平均輝度値をmフレーム分取って，平均輝度値の標準偏差を計算し大きければ，
//mフレーム内の時間経過で動きが激しいと考え，透かしを強く埋め込む．
////そうでない場合は，透かしを比較的弱く埋め込む．
//
//透かしビットが0の時は，平均輝度値の標準偏差を縮小させるように埋め込み，
//透かしビットが1の時は，拡大させるように埋め込む

#define THRESHOLD_DIFF_PIXEL 7

#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <stdio.h>
#include <numeric>
#include <complex>
#include <windows.h>
using comp = std::complex<double>;


#define PROJECT_LOOP 5

const std::string log_file("C:/Users/youhei/Desktop/research_all/research_data/log_all.txt");
const std::string embed_file("C:/Users/youhei/Desktop/research_all/research_data/m2_embednum16x9_one16.txt"); 
const std::string cosine_file("C:/Users/youhei/Desktop/research_all/research_data/cosine_table.txt");

//parameter

const std::string basis_read_file("C:/IHC_videos/xxx");
const std::string basis_write_file("C:/Users/youhei/Desktop/research_all/research_data/mp4_embedded_videos/motion_vector/me_xxx_d3");
const std::string basis_motion_vector_file("C:/share_ubuntu/output/xxx_d1.csv");       // 動きベクトルの元ファイル
const int num_embedframe = 20; //1回当たりの処理で埋め込むフレーム数(偶数)
const double delta = 3; //埋め込み強度

const int BG_width = 16;  // ブロック群の横の長さ
const int BG_height = 9;  // ブロック群の縦の長さ
const int block_width = 8; // DCTブロックの横幅
const int block_height = 8; // DCTブロックの縦幅

const int FRAME_width = 1920;  // フレームの横の長さ
const int FRAME_height = 1080; // フレームの縦の長さ

const int block_size = 8;  //  DCTブロックのサイズ
const int motionvector_block_size = 16; // 動きベクトルのグリッドブロックのサイズ




class mv_class {    // 元データをとりあえず整理して格納する用のクラス
public:
	int frame_index;   // フレームの番号
	cv::Mat x_vector;
	cv::Mat y_vector;
};


class next_pos_all {        //各座標ごとに次のフレームでの座標を格納するクラス
public:
	cv::Mat now_x;
	cv::Mat now_y;
};


// prototype
void init_me(cv::VideoCapture* cap, std::vector<char>* embed, cv::Size* size, std::ofstream* ofs, cv::VideoWriter* writer, std::string read_file, std::string write_file, std::string motion_vector_file, int num_embedframe);
extern std::vector<char> set_embeddata(const std::string filename);
extern cv::VideoCapture capture_open(const std::string read_file);
extern cv::VideoWriter writer_open(const std::string write_file, cv::VideoCapture cap);
extern void motion_embedder(std::vector<cv::Mat>& luminance, std::vector<cv::Mat> &dst_luminance, std::vector<char> embed, int cframe, int num_embedframe, int delta,std::string motion_vector_file, std::vector<mv_class>& mv_all);
extern void operate_lumi_for_zero(std::vector<float> &lumi, float average, float variance, int delta);
void operate_lumi_for_one(std::vector<float> &lumi, float average, float variance, int delta);
cv::VideoWriter mp4_writer_open(const std::string write_file, cv::VideoCapture cap);

// add
void set_motionvector(const std::string motion_vector_file, std::vector<mv_class>& mv_all, int cframe);
std::pair<int, int > get_back_pos(std::vector<mv_class>& mv_all, int frame, int y, int x);
bool Is_there_mv(std::vector<mv_class> &mv_all, int frame);

// common
extern void frame_check(cv::Mat& frame_BGR);      // フレームのエラー処理など
extern void log_write(std::string read_file, std::string write_file);
extern void str_checker(std::string read_file, std::string write_file);
extern bool overwrite_check(std::string write_file);

void change_filename(std::string& read_file, std::string& write_file, std::string& motion_vector_file, int loop_count);

#endif
