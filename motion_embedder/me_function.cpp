#include "me_header.h"

int delta_thisfile = 0;
float average_thisfile = 0;


static double cosine_table[block_height][block_width];  // DCT�ϊ��p�̃R�T�C���e�[�u��

void init_me(cv::VideoCapture* cap, std::vector<char>* embed, cv::Size* size, std::ofstream* ofs, cv::VideoWriter* writer, std::string read_file, std::string write_file, int num_embedframe) {
		*embed = set_embeddata(embed_file);    
		*cap = capture_open(read_file);        
	//	*writer = mp4_writer_open(write_file + ".mp4", *cap);  // mp4�Ȃ̂Ńf�[�^�ʂ����������ߕ����̕K�v�͂Ȃ��D�D
		*writer = writer_open(write_file + ".avi", *cap);
		size->width = cap->get(CV_CAP_PROP_FRAME_WIDTH);
		size->height = cap->get(CV_CAP_PROP_FRAME_HEIGHT);

}

void set_ctable() {    //DCT�ϊ��Ŏg���e�[�u���������ݒ�
	std::ifstream ifs;
	ifs.open(cosine_file);
	if (ifs.fail()) {
		std::cerr << "can't open embed data file\n";
		std::getchar();
		exit(3);
	}

	for (int i = 0; i < block_height; i++) {
		for (int j = 0; j < block_width; j++) {
			cosine_table[i][j] = ifs.get();
		}
	}
}

std::vector<char> set_embeddata(const std::string filename) {
	std::vector<char> embed;
	std::ifstream ifs;
	ifs.open(filename);
	if (ifs.fail()) {
		std::cerr << "can't open embed data file\n";
		std::getchar();
		exit(3);
	}

	while (!ifs.eof())
		embed.push_back(ifs.get());

	return embed;
}

cv::VideoCapture capture_open(const std::string read_file) {
	cv::VideoCapture cap(read_file);
	if (!cap.isOpened()) {
		std::cout << "can't open video file.\n";
		std::getchar();
		exit(4);
	}
	return cap;
}

cv::VideoWriter writer_open(const std::string write_file, cv::VideoCapture cap) {
	cv::VideoWriter writer;
	cv::Size size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	writer.open(write_file, CV_FOURCC('D', 'I', 'B', ' '), cap.get(CV_CAP_PROP_FPS), size);
	if (!writer.isOpened())
		exit(5);
	return writer;
}


cv::Mat filter(cv::Mat luminance) {                           // �u���b�N���̋P�x�l���Ȃ炷(�P�x�l�̕��ω�)
	cv::Mat dst_luminance(luminance.size(), CV_32F);
	int x = 0, y = 0, m = 0, n = 0;
	for (m = 0; m < luminance.cols; m += block_width) {        //  block���Ƃɏ���������
		for (n = 0; n < luminance.rows; n += block_height) {
			float mean = 0, sum = 0;
			for (x = m; x < m + block_width; x++) {             // �u���b�N���̍��v�P�x�l�����߂�
				for (y = n; y < n + block_height; y++) {
					sum += (float)luminance.at<uchar>(y, x);
				}
			}

			mean = sum / (block_width * block_height);        // ��L�̍��v�P�x�l�̕��ϒl
			for (x = m; x < m + block_width; x++) {
				for (y = n; y < n + block_height; y++) {
					dst_luminance.at<float>(y, x) = mean;
				}
			}
		}
	}
	return dst_luminance;
}

float median(std::vector<float> v) {     // �����l��Ԃ�
	int size = v.size();
	std::sort(v.begin(), v.end());

	if (size % 2 == 1) {
		return v[(size - 1) / 2];
	}
	else {
		return (v[(size / 2) - 1] + v[size / 2]) / 2;
	}
}

//void motion_detect(const cv::Mat& p_luminance,const cv::Mat& c_luminance, std::vector<cv::Mat>& check_lumi_array, int cframe, int c_num_embedframe) {
//	// �A������2�t���[���̋P�x�l����
//
//	// �v�f��cv::Mat�^�̔z����w���|�C���^���󂯎���Ă�������Ƃɓ������o���s��
//	cv::Size size(FRAME_WIDTH, FRAME_HEIGHT);
//	cv::Mat lumi_diff(size, CV_8UC3);
//
//	for (int j = 0; j < FRAME_HEIGHT; j += block_height) {
//		for (int i = 0; i < FRAME_WIDTH; i += block_width) {
//			// ��DAAD�ϊ�����ƂȂ������ׂẲ�f�̋P�x�l���啝�ɏオ��̂ŁC�S�Ẳ�f�̒��ōł��P�x�l�̕ω������������̂̕ω��ʂ𑫂����킹�Ă�������
//
//			for (int k = 0; k < block_height; k++) {
//				lumi_diff = std::abs(p_luminance.at<unsigned char>(j, i) - c_luminance.at<unsigned char>(j, i));  // �O��̃t���[���̋P�x�������
//
//			}
//			if ((lumi_diff >= THRESHOLD_DIFF_PIXEL) && (lumi_diff <= 100)) {
//				check_lumi_array[c_num_embedframe].at<unsigned char>(j, i) = 1 + 4 * lumi_diff / 100;
//			}
//			else if (lumi_diff > 100) {
//				check_lumi_array[c_num_embedframe].at<unsigned char>(j, i) = 5;
//			}
//			else {
//				check_lumi_array[c_num_embedframe].at<unsigned char>(j, i) = 1;
//			}
//		}
//	}
//}

void motion_embedder(std::vector<cv::Mat>& luminance, std::vector<cv::Mat> &dst_luminance,std::vector<char> embed, int cframe,int num_embedframe, int delta) {
	std::vector<cv::Mat> means;  //�u���b�N�P�ʂ̕��ϋP�x�l��ێ�
	std::vector<cv::Mat> deviations;  //�u���b�N�P�ʂ̕��ϒl����̕΍���ێ�
	cv::Mat m_means = cv::Mat::zeros(1920, 1080, CV_32F);  //m�t���[���Ԃł́u�u���b�N�P�ʂ̕��ϒl�v�̕��ϒl��ێ�
	std::vector<cv::Mat> m_deviations;  //m�t���[���Ԃł́u�u���b�N�P�ʂ̕��ϒl�v�̕��ϒl����̕΍���ێ�
	int x, y;

	for (int i = 0; i < num_embedframe; i++) {    //  �e��f�̕΍��ƃu���b�N�����ϋP�x�l�����߂�
		cv::Mat temp_luminance = luminance[i].clone();
		means.push_back(filter(temp_luminance));    // luminance�̑傫����CV_32F�̔z�񂪊i�[�����
		temp_luminance.convertTo(temp_luminance, CV_32F);
		deviations.push_back(temp_luminance - means[i]);      // �e��f�̕΍� = �e��f�̋P�x�l-�u���b�N�����ϋP�x�l 
	}

	//m�t���[���ŕ��ς��Ƃ�
	m_means = means[0].clone() / num_embedframe;   // means�̌^�������I�Ɍ��߂����������炱�̕�����for����O����?
	for (int i = 1; i < num_embedframe; i++) {
		m_means += means[i] / num_embedframe;
	}

	//m�t���[���Ԃł́u�u���b�N�P�ʂ̕��ϒl�v�̕��ϒl��ێ�
	std::vector<cv::Mat> t_variance(num_embedframe);
	cv::pow((means[0] - m_means), 2, t_variance[0]);
	cv::Mat variance = t_variance[0].clone();
	for (int i = 1; i < num_embedframe; i++) {
		cv::pow((means[i] - m_means), 2, t_variance[i]);
		variance += t_variance[i];
	}
	variance /= num_embedframe;
	

	//���ߍ��ݏ���
	cv::Mat p_cnt = cv::Mat::zeros(cv::Size(1920,1080), CV_8UC1), m_cnt = cv::Mat::zeros(cv::Size(1920, 1080), CV_8UC1);
	cv::Mat p_cnt0 = cv::Mat::zeros(cv::Size(1920, 1080), CV_8UC1), m_cnt0 = cv::Mat::zeros(cv::Size(1920, 1080), CV_8UC1);
	
	int v_temp;
	float now_point;   // ���݂̍��W�������Ă���DCT�u���b�N�̕��ϋP�x�l
	std::vector<float> lumi(20,0);

	for (x = 0; x < FRAME_WIDTH; x++) {
		for (y = 0; y < FRAME_HEIGHT; y++) {
			v_temp = variance.at<float>(y, x);
			// �u���b�N���͓����������r�b�g�𖄂ߍ��ށD����u���b�N�Q���̈قȂ�u���b�N���̉�f���C�����l�������Ƃ͂Ȃ�
			float num = (embed[(x / block_width) % BG_width + ((y / block_height) % BG_height)*BG_width] == '0') ? 0 : 1;

			if (num == 0) {  // �������r�b�g��0�̎�
				for (int i = 0; i < num_embedframe; i++) {
					lumi[i] = means[i].at<float>(y, x);
				}

				for (int t_delta = delta; t_delta >= 1; t_delta--) {
					if (v_temp >= t_delta * t_delta) {
						operate_lumi(lumi, m_means.at<float>(y,x), v_temp, t_delta);
					}
				}

				for (int i = 0; i < num_embedframe; i++) {
					means[i].at<float>(y, x) = lumi[i];
					lumi[i] = 0;
				}
			}
			else {    // �������r�b�g��1�̎�
				for (int i = 0; i < num_embedframe; i++) {
					now_point = means[i].at<float>(y, x);

					//�ߎ��I�ɒ����l�����������ǂ����𔻒�  (���ϒl�Ɣ�r���Ă���悤�����C���ꂪ���������l�Ƃ̔�r�ɂȂ�̂��H)
					if ((now_point >= m_means.at<float>(y, x) && p_cnt.at<unsigned char>(y, x) != num_embedframe / 2) || m_cnt.at<unsigned char>(y, x) == num_embedframe / 2) {
						now_point += delta;  //�������t���[���̐�
						p_cnt.at<unsigned char>(y, x)++;
					}
					else {
						now_point -= delta;  //-�������t���[���̐�
						m_cnt.at<unsigned char>(y, x)++;
					}

					means[i].at<float>(y, x) = now_point;
				}
			}

		}
	}

	//���ߍ��݌�t���[����Ԃ�
	for (int i = 0; i < num_embedframe; i++) {
		dst_luminance.push_back(deviations[i] + means[i]);
		dst_luminance[i].convertTo(dst_luminance[i], CV_8UC1);
		//cv::imshow("0", dst_luminance[i]);
		//cv::waitKey(200);	
	}
}

bool is_less_than(float i) {
	return ((i < average_thisfile) == 1);
}

bool is_more_than(float i) {
	return ((i > average_thisfile) == 1);
}

void operate_lumi(std::vector<float> &lumi, float average, float variance, int delta) {  // ���ς��ێ����A�W���΍��𕪎U�����ɂ���֐�
	// average, variance��lumi�ŗ^������P�x�l�̕��ςƕ��U�ł���Adelta�͖��ߍ��݋��x
	size_t index_max, index_min; // �ő�A�ŏ��̗v�f�̓Y����
	size_t num_low_ave = 0;  // ���ς����Ⴂ��
	size_t num_high_ave = 0; //���ς���������
	float now_variance;
	std::vector<double> temp_lumi(20);

	average_thisfile = average;

	for (int i = 0; i < end(lumi) - begin(lumi); i++) {
		temp_lumi[i] = lumi[i];
	}

	for (int limit_time  = 0; limit_time < 30; limit_time++) {  // �Ȃ�30�H
		// ���ς���ł������v�f�̃C���f�b�N�X�����߂�
		std::vector<double>::iterator itr_max = std::max_element(temp_lumi.begin(), temp_lumi.end());
		std::vector<double>::iterator itr_min = std::min_element(temp_lumi.begin(), temp_lumi.end());
		index_max = std::distance(temp_lumi.begin(), itr_max);
		index_min = std::distance(temp_lumi.begin(), itr_min);
			   
		num_low_ave = std::count_if(temp_lumi.begin(), temp_lumi.end(), is_less_than);
		num_high_ave = std::count_if(temp_lumi.begin(), temp_lumi.end(), is_more_than);
		now_variance = 0;

		temp_lumi[index_max]--;
		for (int j = 0; j < (end(temp_lumi) - begin(temp_lumi)); j++) {
			if (temp_lumi[j] < average) {
				temp_lumi[j] += 1 / num_low_ave;
			}
		}


		temp_lumi[index_min]++;
		for (int j = 0; j < (end(temp_lumi) - begin(temp_lumi)); j++) {
			if (temp_lumi[j] > average) {
				temp_lumi[j] -= 1 / num_high_ave;
			}
		}

		for (int k = 0; k < (end(temp_lumi) - begin(temp_lumi)); k++) {
			now_variance += (temp_lumi[k] - average) * (temp_lumi[k] - average);
		}
		

		if ((now_variance <= (variance * (10 - delta) / 10)) || (now_variance <= (variance - delta * delta))) { 
			break;
		}
	}

	for (int i = 0; i < end(lumi) - begin(lumi); i++) {
		lumi[i] = temp_lumi[i];  
	}
}
