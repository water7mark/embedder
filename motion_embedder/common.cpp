#include "me_header.h"

// ���̃t�@�C���͂����ꃉ�C�u���������ׂ���
void frame_check(cv::Mat& frame_BGR) {
	if (frame_BGR.empty()) {  // �Ӗ��͂���̂�?
		exit(112);
	}
	if (frame_BGR.size() != cv::Size(1920, 1080)) {
		cv::resize(frame_BGR, frame_BGR, cv::Size(1920, 1080));
	}

	return;
}

void log_write(std::string read_file, std::string write_file) {
	char date[64];
	std::ofstream ofs(log_file, std::ios::app);   // �㏑��
	time_t t = time(NULL);
	tm now;
	errno_t error;

	if (!ofs)
	{
		std::cout << "�t�@�C�����J���܂���ł����B" << std::endl;
		std::cin.get();
		exit(9);
	}

	error = localtime_s(&now, &t);

	strftime(date, sizeof(date), "%Y/%m/%d %a %H:%M:%S\n", &now);
	ofs << date << "embedder::" << "read_file�E" << read_file << "\n" << "write_file�E" << write_file << "\n" << embed_file << "\n" << std::endl;
	ofs << "----------------------------------------------\n";
	std::cout << log_file << "�ɏ������݂܂����B" << std::endl;

}


void str_checker(std::string read_file, std::string write_file) {
	// �{���́C�A�z�z����g���ׂ��ł�?   // ���������͊֐������ׂ�
	std::vector<std::string> r_label = { "Basket","Library","Lego", "Walk1", "Walk2" };
	std::vector<std::string> w_label = { "basket","library","lego", "walk1", "walk2" };
	std::vector<std::string> m_array = { "m10", "m20", "m30" ,"m40" };
	std::vector<std::string> delta_array = { "d1" "d2", "d3" ,"d5", "d10" };


	if (write_file.find("test") != std::string::npos) {
		return;
	}

	// read_file , write_file check
	for (int i = 0; i < end(r_label) - begin(r_label); i++) {
		if (read_file.find(r_label[i]) != std::string::npos) {     // ���Y�����񂪊܂܂�Ă���
			if (write_file.find(w_label[i]) != std::string::npos) { // write_file�ɂ����������񂪊܂܂�Ă���
				break;
			}
			else {   // ���������񂪊܂܂�Ă��Ȃ��D�~�X���Ă�ꍇ
				std::cout << "error: read label is not equal write label" << std::endl;
				std::cout << "read_file:" << read_file << std::endl;
				std::cout << "write_file:" << write_file << std::endl;
				getchar();
				exit(0);
			}
		}
		else if (i == end(r_label) - begin(r_label) - 1) {
			std::cout << "error: wrong read_file name!!!!" << std::endl;
			getchar();
			exit(1);
		}
	}
}

bool overwrite_check(std::string write_file) {       // ��������f�[�^���㏑�����Ȃ��悤�ɂ��邽�߂̃`�F�b�N�֐�
	std::string new_write_file = write_file + ".mp4";
	std::ifstream ifs(new_write_file);

	if (ifs.is_open()) {
		std::cout << "error: overwrite::" << write_file << std::endl;
		getchar();
		return false;
	}

	return true;
}

void change_filename(std::string& read_file,  std::string& write_file, int loop_count) {
	const std::string mp4_read_array[5] = { "Basket.mp4", "Library.mp4", "Lego.mp4", "Walk1.mp4", "Walk2.mp4" };
	const std::string read_array[5] = { "basket" , "library" , "lego", "walk1", "walk2" };
	int change_point = 0;

	// read_file�̑���(loop_count�ɉ����ăt�@�C���̖��O�̓���̃^�C�g��������ύX)
	change_point = (int)read_file.find("xxx");
	read_file.replace(change_point, 3, mp4_read_array[loop_count - 1]);

	// write_file�̑���
	change_point = (int)write_file.find("xxx");
	write_file.replace(change_point, 3, read_array[loop_count - 1]);

	// �R���\�[���o��
	std::cout << read_file  << std::endl;
	std::cout << write_file << std::endl;
}