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
const std::string basis_motion_vector_file("C:/share_ubuntu/output/xxx_d1.csv");       // �����x�N�g���̌��t�@�C��
const int num_embedframe = 20; //1�񓖂���̏����Ŗ��ߍ��ރt���[����(����)
const double delta = 3; //���ߍ��݋��x

const int BG_width = 16;  // �u���b�N�Q�̉��̒���
const int BG_height = 9;  // �u���b�N�Q�̏c�̒���
const int block_width = 8; // DCT�u���b�N�̉���
const int block_height = 8; // DCT�u���b�N�̏c��

const int FRAME_width = 1920;  // �t���[���̉��̒���
const int FRAME_height = 1080; // �t���[���̏c�̒���

const int block_size = 8;  //  DCT�u���b�N�̃T�C�Y
const int motionvector_block_size = 16; // �����x�N�g���̃O���b�h�u���b�N�̃T�C�Y




class mv_class {    // ���f�[�^���Ƃ肠�����������Ċi�[����p�̃N���X
public:
	int frame_index;   // �t���[���̔ԍ�
	cv::Mat x_vector;
	cv::Mat y_vector;
};


class next_pos_all {        //�e���W���ƂɎ��̃t���[���ł̍��W���i�[����N���X
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
extern void frame_check(cv::Mat& frame_BGR);      // �t���[���̃G���[�����Ȃ�
extern void log_write(std::string read_file, std::string write_file);
extern void str_checker(std::string read_file, std::string write_file);
extern bool overwrite_check(std::string write_file);

void change_filename(std::string& read_file, std::string& write_file, std::string& motion_vector_file, int loop_count);

#endif
