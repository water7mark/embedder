// �O�̃v���O������temp01�̃��|�W�g���ɓ���Ă܂�


#include "me_header.h"

int delta_thisfile = 0;
float average_thisfile = 0;


#define dx 0
#define dy 1

static double cosine_table[block_height][block_width];  // DCT�ϊ��p�̃R�T�C���e�[�u��
static int matrix_height = 134;
static int matrix_width = 120;

#define MV_DETA_SIZE CV_8SC1
#define NP_DETA_SIZE CV_16SC1

#define MV_DETA_TYPE char
#define NP_DETA_TYPE short


void init_me(cv::VideoCapture* cap, std::vector<char>* embed, cv::Size* size, std::ofstream* ofs, cv::VideoWriter* writer, std::string read_file, std::string write_file, std::string motion_vector_file, int num_embedframe) {
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

cv::VideoWriter mp4_writer_open(const std::string write_file, cv::VideoCapture cap) {
	cv::VideoWriter writer;
	cv::Size size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	writer.open(write_file, CV_FOURCC('M', 'P', '4', 'V'), cap.get(CV_CAP_PROP_FPS), size);
	if (!writer.isOpened())
		exit(5);
	return writer;
}

void set_motionvector(const std::string motion_vector_file, std::vector<mv_class>& mv_all, int cframe) {
	// �t�@�C�����̃f�[�^���c�傷���邽�߁A���ߍ��ݎ��ɓK�X���̊֐����Ăяo���āA���̓s�x�t�@�C������f�[�^��ǂݏo��
	// ���ߍ��ސ擪�t���[��(cframe)�܂Ńt�@�C���ǂݍ��݂��΂��Ă���num_embedframe���f�[�^���擾����

	std::ifstream ifs;
	ifs.open(motion_vector_file);
	if (ifs.fail()) {
		std::cerr << "can't open motion vector data file\n";
		std::getchar();
		exit(3);
	}

	std::string str;
	std::string cframe_str = std::to_string(cframe);
	cframe_str.insert(0, "pts=");
	while (str.find(cframe_str) == std::string::npos) {   //pts=cframe�ƂȂ�܂Ŕ�΂�
		getline(ifs, str, ' ');
	}


	// ���������K�{
	mv_class temp_class;
	// ������
	temp_class.frame_index = -1;
	temp_class.x_vector = cv::Mat::zeros(cv::Size(FRAME_width / motionvector_block_size, FRAME_height / motionvector_block_size), MV_DETA_SIZE);
	temp_class.y_vector = cv::Mat::zeros(cv::Size(FRAME_width / motionvector_block_size, FRAME_height / motionvector_block_size), MV_DETA_SIZE);

	std::vector<int> debug_array(120);// debug�p

	for (int pts = 1; pts < num_embedframe + 1 && !ifs.eof();) {
		int temp_start;
		std::string temp_str;
		int temp_count = 0;


		getline(ifs, str, ' ');

		//if (str.find("pts=9") != std::string::npos) {  // debug�p
		//	int abc;
		//	abc = 1;
		//}

		if (str.find("pts") != std::string::npos) {          // pts���o��΁A1�t���[���Ƃ݂Ȃ� 
			pts++;
		}
		else if (str.find("frame_index") != std::string::npos) {
			temp_start = str.find("frame_index");
			temp_str = str.substr(temp_start + 12, str.length());        //�����x�N�g���t�@�C�����ł́Aframe_index=�����ɂȂ��Ă���Ƃ����O��
			if (atoi(temp_str.c_str()) == -1) {
				temp_class.frame_index = atoi(temp_str.c_str());
			}
			else {
				temp_class.frame_index = pts - 1 + cframe;
			}
		}
		else if (str.find("shape") != std::string::npos) {
			temp_start = str.find("shape");

			for (int i = 0; i < 120 * matrix_height; i++) {
				getline(ifs, str, '\t');

				if (i == 0) {
					temp_str = str.substr(13, str.length() - 13);             //���̍s���v���ȁE�E�E�E
				}
				else if (i % 120 == 0) {   // �s�̐؂�ڂ�\n���폜
					temp_str = str.substr(1, str.length() - 1);              // ���̍s�_�C�W���u���ȁE�E
				}
				else {
					temp_str = str;
				}

				if (i < 120 * matrix_height / 2) {
					temp_class.x_vector.at<MV_DETA_TYPE>(i / 120, i % 120) = atoi(temp_str.c_str());
				}
				else {
					temp_class.y_vector.at<MV_DETA_TYPE>((i / 2) / 120, (i / 2) % 120) = atoi(temp_str.c_str());
				}
			}

			//int debug_num = 0;
			//if (pts == 10) {
			//	debug_num = temp_class.x_vector.at<MV_DETA_TYPE>(0, 19);
			//	std::cout << temp_class.x_vector.at<MV_DETA_TYPE>(0, 19) << std::endl;
			//}
			
			// �[���R�s�[
			mv_all[pts - 1].frame_index = temp_class.frame_index;
			mv_all[pts - 1].x_vector = temp_class.x_vector.clone();
			mv_all[pts - 1].y_vector = temp_class.y_vector.clone();


			// �ēx������
			temp_class.frame_index = -1;
			temp_class.x_vector = cv::Mat::zeros(cv::Size(FRAME_width / motionvector_block_size, FRAME_height / motionvector_block_size), MV_DETA_SIZE);
			temp_class.y_vector = cv::Mat::zeros(cv::Size(FRAME_width / motionvector_block_size, FRAME_height / motionvector_block_size), MV_DETA_SIZE);

			//if (pts == 10) {
			//	debug_num = mv_all[pts - 1].x_vector.at<MV_DETA_TYPE>(0, 19);
			//	std::cout << mv_all[pts - 1].x_vector.at<MV_DETA_TYPE>(0, 19) << std::endl;
			//}
		}
	}

	//for (int i = 0; end(debug_array) - begin(debug_array); i++) {
	//	std::cout << debug_array[i] << std::endl;
	//}
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

void motion_embedder(std::vector<cv::Mat>& luminance, std::vector<cv::Mat> &dst_luminance, std::vector<char> embed, int cframe, int num_embedframe, int delta, std::string motion_vector_file, std::vector<mv_class>& mv_all) {
	std::vector<cv::Mat> means;  //�u���b�N�P�ʂ̕��ϋP�x�l��ێ�
	std::vector<cv::Mat> deviations;  //�u���b�N�P�ʂ̕��ϒl����̕΍���ێ�
	cv::Mat m_means = cv::Mat::zeros(FRAME_width, FRAME_height, CV_32F);  //m�t���[���Ԃł́u�u���b�N�P�ʂ̕��ϒl�v�̕��ϒl��ێ�
	std::vector<cv::Mat> m_deviations;  //m�t���[���Ԃł́u�u���b�N�P�ʂ̕��ϒl�v�̕��ϒl����̕΍���ێ�
	int x, y;

	// �t�@�C������f�[�^�擾
	set_motionvector(motion_vector_file, mv_all, cframe);

	// debug
	//for (int i = 0; i < 1; i++) {
	//	for (int k = 0; k < block_size; k++) {
	//		for (int l = 0; l < block_size; l++) {
	//			std::cout << +luminance[i].at<unsigned char>(k, l) << std::endl;
	//		}
	//	}
	//}


	// means, m_means , deviations, variance �̍쐬
	for (int i = 0; i < num_embedframe; i++) {    //  �e��f�̕΍��ƃu���b�N�����ϋP�x�l�����߂�
		cv::Mat temp_luminance = luminance[i].clone();
		means.push_back(filter(temp_luminance));
		temp_luminance.convertTo(temp_luminance, CV_32F);
		deviations.push_back(temp_luminance - means[i]);      // �e��f�̕΍� = �e��f�̋P�x�l-�u���b�N�����ϋP�x�l 
	}

	//m�t���[���Ԃł́u�u���b�N�P�ʂ̕��ϒl�v�̕��ϒl��ێ�
	m_means = means[0].clone() / num_embedframe;
	for (int i = 1; i < num_embedframe; i++) {
		m_means += means[i] / num_embedframe;
	}

	std::vector<cv::Mat> t_variance(num_embedframe);
	cv::pow((means[0] - m_means), 2, t_variance[0]);
	cv::Mat variance = t_variance[0].clone();
	for (int i = 1; i < num_embedframe; i++) {
		cv::pow((means[i] - m_means), 2, t_variance[i]);
		variance += t_variance[i];
	}
	variance /= num_embedframe;
	// �쐬�I��


	std::vector<cv::Mat>Is_stop(20, cv::Mat::zeros(cv::Size(FRAME_width/ 8, FRAME_height/ 8), CV_8UC1));     // ���Y��f���㏑������邩����C�㏑�����ꂽ�t���[����-1�C���̉�f���Ăё��̉�f�ֈړ������Ȃ炻�̃t���[����1���i�[����
    std::vector<cv::Mat>Is_move(20, cv::Mat::zeros(cv::Size(FRAME_width /8, FRAME_height / 8), CV_8UC1)); // ���Y��f�����̉�f�ʒu�ɓ������ǂ����̔���ɗp����D20�t���[���̂ǂ����ňړ�����ꍇ��true�ɂȂ�A��x���ړ����Ȃ��Ȃ�false

	std::vector<cv::Mat> result_lumi(20, cv::Mat::zeros(cv::Size(FRAME_width / 8, FRAME_height/ 8), CV_32F));
	std::vector<cv::Mat> lumi_map(20, cv::Mat::zeros(cv::Size(FRAME_width / 8, FRAME_height / 8), CV_32F));    // means��32F�ł��邩��^�����킹��D
	std::vector<cv::Mat> comp(20, cv::Mat::zeros(cv::Size(FRAME_width /8, FRAME_height/8), CV_8UC1));        // �v�Z���ʂ�result�Ɋi�[����Ă���Ȃ��[����Ԃ�

	// ��ɓ����x�N�g���̏���
	for (int i = num_embedframe - 1; i > 0 ; i--)        // ���̕ӂ̗v�f���Ȃǂ̌��ˍ�����������x�m���߂悤 2020 1_ 17
		for (y = 0; y < FRAME_height / block_size; y++) {
			for (x = 0; x < FRAME_width / block_size; x++) {{
					if (Is_there_mv(mv_all, cframe + i)) {  // ���݂̃t���[���ԍ���^����Ɠ����x�N�g�����o�͂���Ă��邩�Ԃ��֐�	
						std::pair<int, int> back_pixel = get_back_pos(mv_all, cframe + i, y, x);    //�t���[���ԍ���pts����������ɂȂ��Ă��Ȃ����m�F����(���݂̃t���[����Ԃ��΂����Ǝv���)

						Is_stop[i].at<unsigned char>(y, x) = 1;
						Is_stop[i - 1].at<unsigned char>(back_pixel.first, back_pixel.second) = -1;
						Is_move[i].at<unsigned char>(y, x) = 1;
					}
					else {
						// �p�������𖞂����Ȃ��悤�ɂ���for��2������
						y = FRAME_height / block_size - 1;
						x = FRAME_width / block_size - 1;
					}
				}
			}
	}

	//���r���ŏ㏑�����ꂽ�Ƃ��Ă����̏��́Clumi_map�ɂ͍ڂ�Ȃ�(�ł��C�O���̃��[�v��num_embedframe�ɂ���΁C)


	// 20�t���[���̍Ōォ�珇�ɓ����x�N�g���𒲂ׂ�20�̋P�x���m�ۂ���
	int num;          // ���݂̉�f�Ɋ��蓖�Ă�ׂ��������r�b�g���i�[
	// lumi_map���߂�
	for (y = 0; y < FRAME_height / block_size; y++) {
		for (x = 0; x < FRAME_width / block_size; x++) {
			int temp_y = y;
			int temp_x = x;

			lumi_map[num_embedframe - 1].at<float>(temp_y, temp_x) = means[num_embedframe - 1].at<float>(temp_y *  block_size, temp_x *  block_size);
			comp[num_embedframe- 1].at<unsigned char>(temp_y, temp_x) = 1;


			int jump_flg = 0;   // �ړ������ۂɈړ����Isstop��-1������D���̂��ߍŏ���if�Œʂ�������Ɏ��̃��[�v��elseif�Ɉ���������Ȃ��悤�ɂ��邽��
			for (int i = num_embedframe - 1; i > 0; i--) {
				if (Is_stop[i].at<unsigned char>(temp_y, temp_x) == 1) {     // ���̉�f�ʒu�Ɉړ����Ă���Ȃ�
					std::pair<int, int> back_point;     
					back_point = get_back_pos(mv_all, cframe + i , temp_y, temp_x);
					lumi_map[i - 1].at<float>(y, x) = means[i - 1].at<float>(back_point.first *  block_size, back_point.second *  block_size);
					comp[i - 1].at<unsigned char>(back_point.first, back_point.second) = 1;
					temp_y = back_point.first;
					temp_x = back_point.second;
					jump_flg = 1;
				}
				else if (Is_stop[i].at<unsigned char>(temp_y, temp_x) == -1 && jump_flg != 1) {       // �ړ������㏑�����ꂽ�Ȃ�lumi_map�ɂ͉����ǂݍ��܂Ȃ�
					continue;
				}
				else { 
					lumi_map[i - 1].at<float>(y, x) = means[i - 1].at<float>(temp_y *  block_size, temp_x *  block_size);
					comp[i - 1].at<unsigned char>(temp_y, temp_x) = 1;
					jump_flg = 0;
				}
			}
		}
	}


	// ���ǂ��̒i�K��lumi_map�͂ǂ��Ȃ��Ă���̂��C���ׂĖ��܂��Ă���̂��C��������̂��D�D�D
	// �����ׂĖ��܂��Ă���D(�����x�N�g���t�@�C���͂��ׂẴu���b�N�ɂ��ĉ��炩�̐��l�����蓖�Ă��Ă���C�����get_next_pos�ŗ��p���Ă���̂����瓖�R)
	// lumi_map�ɂ́Clumi_map�̉�f�ʒu���Ƃɐ擪�t���[�����炻�̃u���b�N���ǂ̂悤�Ɉړ���������������Ă���
	// ������������͓̂��������������Ƃ��̂�

	std::vector<float> lumi(20, 0);         //�@lumi_map�̊e�u���b�N�̋P�x�l�����o���āC�v�Z����ۂ�float�ɂ���K�v������
	int sum_stop;
	float ave_lumi = 0;   
	float var_lumi = 0;

	for (y = 0; y < FRAME_height / block_size; y ++) {
		for (x = 0; x < FRAME_width / block_size; x++) {
			num = (embed[(x / block_width) % BG_width + ((y / block_height) % BG_height)*BG_width] == '0') ? 0 : 1;
			int temp_y = y;
			int temp_x = x;

			for (int i = num_embedframe - 1; i >= 0; i--) {
				lumi[i] = lumi_map[i].at<float>(y, x);
				sum_stop = Is_stop[i].at<unsigned char>(temp_y, temp_x);       // �������āC�ړ�������ŏ㏑������Ă����v�H�H���ƂŊm�F
				ave_lumi += lumi[i];
			}
			
			ave_lumi /= num_embedframe;
			for (int i = 0; i < num_embedframe; i++) {
				var_lumi += pow((lumi[i] - ave_lumi), 2);
			}
			var_lumi / num_embedframe;   // ���U�̒�`��:(�v�f-����)^2 /�v�f��

			if (sum_stop < 0) {    //�@�r���ŏ㏑�����ꂽ�ꍇ
				continue;
			}
			else {              // 20�t���[������f������ꍇ
				// �������r�b�g�ɉ����Čv�Z
				if (num == 0) {
					for (int t_delta = delta; t_delta >= 1; t_delta--) {   // delta�����ǂ̒��x���U���傫�����ǂ����ő��삷��ʂ����߂Ă���
						if (var_lumi >= t_delta * t_delta) {
							operate_lumi_for_zero(lumi, ave_lumi, var_lumi, t_delta);
							break;
						}
					}
				}
				else {    // �������r�b�g��1�̎�
					operate_lumi_for_one(lumi, ave_lumi, var_lumi, delta);
				}


				// ���ߍ��񂾌��ʂ�result_lumi�Ɋi�[
				result_lumi[num_embedframe - 1].at<float>(y, x) = lumi[num_embedframe - 1];
				temp_y = y;
				temp_x = x;
				for (int i = num_embedframe - 1; i > 0 ; i--) {
					std::pair<int, int> back_point;
					back_point = get_back_pos(mv_all, cframe + i, temp_y, temp_x);
					result_lumi[i - 1].at<float>(temp_y, temp_x) = lumi[i - 1];
				}
			}
		}
	}
		

	//���ߍ��݌�t���[����Ԃ�
	for (int i = 0; i < num_embedframe; i++) {
		for (y = 0; y < FRAME_height / block_size; y++) {
			for (x = 0; x < FRAME_width / block_size; x++) {
				if (comp[i].at<unsigned char>(y, x) == 0) {
					result_lumi[i].at<float>(y, x) = means[i].at<float>(y * block_size, x *  block_size);
				}
			}
		}

		float temp_debug;
		for (y = 0; y < FRAME_height / block_size; y++) {
			for (x = 0; x < FRAME_width / block_size; x++) {

				for (int m = 0; m < block_height; m++) {
					for (int n = 0; n < block_width; n++) {
						deviations[i].at<float>(y * block_height + m, x * block_width + n) += result_lumi[i].at<float>(y , x);
					}
				}
			}
		}
		dst_luminance.push_back(deviations[i]);
		dst_luminance[i].convertTo(dst_luminance[i], CV_8UC1);
	//	cv::imshow("0", dst_luminance[i]);
		//cv::waitKey(200);	
	}
}

bool is_less_than(float i) {
	return ((i < average_thisfile) == 1);
}

bool is_more_than(float i) {
	return ((i > average_thisfile) == 1);
}

void operate_lumi_for_one(std::vector<float> &lumi, float average, float variance, int delta) {
	size_t index_max, index_min; // �ő�A�ŏ��̗v�f�̓Y����
	size_t num_low_ave = 0;  // ���ς����Ⴂ��
	size_t num_high_ave = 0; //���ς���������
	float now_variance;

	int p_cnt = 0;
	int m_cnt = 0;

	for (int i = 0; i < num_embedframe; i++) {
		if ((lumi[i] >= average && p_cnt != num_embedframe / 2) || m_cnt == num_embedframe / 2) {
			lumi[i] += delta;  //�������t���[���̐�
			p_cnt++;
		}
		else {
			lumi[i] -= delta;  //-�������t���[���̐�
			m_cnt++;
		}
	}
}

void operate_lumi_for_zero(std::vector<float> &lumi, float average, float variance, int delta) {  // ���ς��ێ����A�W���΍��𕪎U�����ɂ���֐�
	// average, variance��lumi�ŗ^������P�x�l�̕��ςƕ��U�ł���Adelta�͖��ߍ��݋��x
	size_t index_max, index_min; // �ő�A�ŏ��̗v�f�̓Y����
	size_t num_low_ave = 0;  // ���ς����Ⴂ��
	size_t num_high_ave = 0; //���ς���������
	float now_variance;
	std::vector<double> temp_lumi(20);

	average_thisfile = average;


	// �v�Z�p�̔z��Ɋi�[
	for (int i = 0; i < end(lumi) - begin(lumi); i++) {
		temp_lumi[i] = lumi[i];
	}


	for (int limit_time = 0; limit_time < 30; limit_time++) {
		// ���ς���ł������v�f�̃C���f�b�N�X�����߂�
		std::vector<double>::iterator itr_max = std::max_element(temp_lumi.begin(), temp_lumi.end());
		std::vector<double>::iterator itr_min = std::min_element(temp_lumi.begin(), temp_lumi.end());
		index_max = std::distance(temp_lumi.begin(), itr_max);
		index_min = std::distance(temp_lumi.begin(), itr_min);

		num_low_ave = std::count_if(temp_lumi.begin(), temp_lumi.end(), is_less_than);
		num_high_ave = std::count_if(temp_lumi.begin(), temp_lumi.end(), is_more_than);
		now_variance = 0;

		temp_lumi[index_max]--;
		temp_lumi[index_min]++;

		for (int k = 0; k < (end(temp_lumi) - begin(temp_lumi)); k++) {
			now_variance += (temp_lumi[k] - average) * (temp_lumi[k] - average);
		}

		if ((now_variance <= (variance * (10 - delta) / 10)) || (now_variance <= (variance - delta * delta))) {
			break;
		}
	}


	// ���ɖ߂�
	for (int i = 0; i < end(lumi) - begin(lumi); i++) {
		lumi[i] = temp_lumi[i];
	}
}



int ptob(int pixel_pos) {
	return pixel_pos * block_size / motionvector_block_size;
}

int btop(int block_pos) {
	return block_pos / motionvector_block_size * motionvector_block_size / block_size;
}


// mv_all[�����̍��W] = �����̍��W - �ߋ��̍��W
std::pair<int, int > get_back_pos(std::vector<mv_class>& mv_all, int frame, int y, int x) {
	// �t���[���ԍ��ƍ��W��^����Ƃ��̎��̍��W��Ԃ�)
	// ���������Ă��Ȃ���Ό��̍��W��Ԃ�
	int bl_y = ptob(y);
	int bl_x = ptob(x);

	std::pair<int, int> back_pos;

	if (motionvector_block_size == 16) {       // �t���[���̏c����1080�̎��̓}�N���u���b�N�T�C�Y16�ł͊���؂ꂸ�C8��f���]��D���̏ꍇ�C���̂܂܍��W��Ԃ�
		if (y == 134) {
			back_pos.first = y;
			back_pos.second = x;

			return back_pos;
		}
	}

	
	int temp_y = mv_all[frame % num_embedframe].y_vector.at<MV_DETA_TYPE>(bl_y, bl_x);
	int temp_x = mv_all[frame % num_embedframe].x_vector.at<MV_DETA_TYPE>(bl_y, bl_x);


	back_pos = std::make_pair(y - btop(temp_y), x - btop(temp_x));


	// ����̃G���[���Ƃ肠�������P���邽��
	if (back_pos.first < 0 || back_pos.first >= (FRAME_height / block_size)) {
		back_pos.first = y;
	}
	if (back_pos.second < 0 || back_pos.second >= (FRAME_width / block_size)) {
		back_pos.second = x;
	}

	return back_pos;
}


bool Is_there_mv(std::vector<mv_class> &mv_all, int frame) {  // �_�~�[�ꍇ��false��Ԃ�

	if (mv_all[frame % num_embedframe].frame_index == -1) {
		return false;
	}
	else {
		return true;
	}
}