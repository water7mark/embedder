#pragma once

#ifndef _ME_HEADER_NAKA_H
#define _ME_HEADER_NAKA_H
// ���̃v���O�����̖ړI�F
// 2018 12 25����
//DCT�u���b�N�̕��ϋP�x�l��m�t���[��������āC���ϋP�x�l�̕W���΍����v�Z���傫����΁C
//m�t���[�����̎��Ԍo�߂œ������������ƍl���C���������������ߍ��ށD
////�����łȂ��ꍇ�́C���������r�I�キ���ߍ��ށD
//
//�������r�b�g��0�̎��́C���ϋP�x�l�̕W���΍����k��������悤�ɖ��ߍ��݁C
//�������r�b�g��1�̎��́C�g�傳����悤�ɖ��ߍ���




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

const std::string log_file("C:/Users/youhei/Desktop/research_all/research_data/log_all.txt");// �����ƈႤ�̂Œ���
const std::string embed_file("C:/Users/youhei/Desktop/research_all/research_data/m1_try00_embednum16x9.txt");  // �I�I�v�m�F�I�I
const std::string cosine_file("C:/Users/youhei/Desktop/research_all/research_data/cosine_table.txt");

//parameter
const std::string basis_read_file("C:/IHC_videos/xxx");
const std::string basis_write_file("C:/Users/youhei/Desktop/research_all/research_data/mp4_embedded_videos/ver3/ver3_1/ver3_1_4/xxx_ver3_1_4");
const int num_embedframe = 20; //1�񓖂���̏����Ŗ��ߍ��ރt���[����(����)
const double delta = 3; //���ߍ��݋��x

const int BG_width = 16;  // �u���b�N�Q�̉��̒���
const int BG_height = 9;  // �u���b�N�Q�̏c�̒���
const int block_width = 8; // DCT�u���b�N�̉���
const int block_height = 8; // DCT�u���b�N�̏c��

const int FRAME_width = 1920;  // �t���[���̉��̒���
const int FRAME_height = 1080; // �t���[���̏c�̒���



// prototype
extern void init_me(cv::VideoCapture* cap, std::vector<char>* embed, cv::Size* size, std::ofstream* ofs, cv::VideoWriter* writer, std::string read_file, std::string write_file, int num_embedframe);
extern std::vector<char> set_embeddata(const std::string filename);
extern cv::VideoCapture capture_open(const std::string read_file);
extern cv::VideoWriter mp4_writer_open(const std::string write_file, cv::VideoCapture cap);

//extern void motion_detect(const cv::Mat& p_luminance, const cv::Mat& c_luminance, std::vector<cv::Mat>& check_lumi_array, int cframe, int c_num_embedframe); // �������o����check_lumi_array�ɓ����
extern void motion_embedder(std::vector<cv::Mat>& luminance, std::vector<cv::Mat> &dst_luminance, std::vector<cv::Mat>& check_array, std::vector<char> embed, int cframe, int num_embedframe, int delta); // 
extern void operate_lumi(std::vector<float> &lumi, float average, float variance, int delta);


// common
extern void frame_check(cv::Mat& frame_BGR);      // �t���[���̃G���[�����Ȃ�
extern void log_write(std::string read_file, std::string write_file);
extern void str_checker(std::string read_file, std::string write_file);
extern bool overwrite_check(std::string write_file);

extern void change_filename(std::string& read_file, std::string& write_file, int loop_count);
//void set_ctable();
//void dct_embedder(std::vector<cv::Mat>& luminance, std::vector<cv::Mat> &dst_luminance, std::vector<char> embed, int cframe, int num_embedframe, int delta);


//void dc_trans(const std::vector<std::vector<double>>& flumi, std::vector <std::vector<double>>& dst_flumi);
//void idct_trans(std::vector<std::vector<double>>& dct_lumi, std::vector<std::vector<double>>& dst_Flumi);

#endif

