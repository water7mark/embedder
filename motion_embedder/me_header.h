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




#define FRAME_WIDTH 1920
#define FRAME_HEIGHT 1080
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

const std::string log_file("C:/Users/youhei/Desktop/research_all/research_data/log_all.txt");// いつもと違うので注意
const std::string embed_file("C:/Users/youhei/Desktop/research_all/research_data/m1_try00_embednum16x9.txt");  // ！！要確認！！
const std::string cosine_file("C:/Users/youhei/Desktop/research_all/research_data/cosine_table.txt");

//parameter
const std::string basis_read_file("C:/IHC_videos/xxx");
const std::string basis_write_file("C:/Users/youhei/Desktop/research_all/research_data/mp4_embedded_videos/ver3/ver3_1/ver3_1_4/xxx_ver3_1_4");
const int num_embedframe = 20; //1回当たりの処理で埋め込むフレーム数(偶数)
const double delta = 3; //埋め込み強度

const int BG_width = 16;  // ブロック群の横の長さ
const int BG_height = 9;  // ブロック群の縦の長さ
const int block_width = 8; // DCTブロックの横幅
const int block_height = 8; // DCTブロックの縦幅

const int FRAME_width = 1920;  // フレームの横の長さ
const int FRAME_height = 1080; // フレームの縦の長さ



// prototype
extern void init_me(cv::VideoCapture* cap, std::vector<char>* embed, cv::Size* size, std::ofstream* ofs, cv::VideoWriter* writer, std::string read_file, std::string write_file, int num_embedframe);
extern std::vector<char> set_embeddata(const std::string filename);
extern cv::VideoCapture capture_open(const std::string read_file);
extern cv::VideoWriter mp4_writer_open(const std::string write_file, cv::VideoCapture cap);

//extern void motion_detect(const cv::Mat& p_luminance, const cv::Mat& c_luminance, std::vector<cv::Mat>& check_lumi_array, int cframe, int c_num_embedframe); // 動き検出してcheck_lumi_arrayに入れる
extern void motion_embedder(std::vector<cv::Mat>& luminance, std::vector<cv::Mat> &dst_luminance, std::vector<cv::Mat>& check_array, std::vector<char> embed, int cframe, int num_embedframe, int delta); // 
extern void operate_lumi(std::vector<float> &lumi, float average, float variance, int delta);


// common
extern void frame_check(cv::Mat& frame_BGR);      // フレームのエラー処理など
extern void log_write(std::string read_file, std::string write_file);
extern void str_checker(std::string read_file, std::string write_file);
extern bool overwrite_check(std::string write_file);

extern void change_filename(std::string& read_file, std::string& write_file, int loop_count);
//void set_ctable();
//void dct_embedder(std::vector<cv::Mat>& luminance, std::vector<cv::Mat> &dst_luminance, std::vector<char> embed, int cframe, int num_embedframe, int delta);


//void dc_trans(const std::vector<std::vector<double>>& flumi, std::vector <std::vector<double>>& dst_flumi);
//void idct_trans(std::vector<std::vector<double>>& dct_lumi, std::vector<std::vector<double>>& dst_Flumi);

#endif

