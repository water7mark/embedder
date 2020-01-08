#include "me_header.h"

// このファイルはいずれライブラリ化すべき
void frame_check(cv::Mat& frame_BGR) {
	if (frame_BGR.empty()) {  // 意味はあるのか?
		exit(112);
	}
	if (frame_BGR.size() != cv::Size(1920, 1080)) {
		cv::resize(frame_BGR, frame_BGR, cv::Size(1920, 1080));
	}

	return;
}

void log_write(std::string read_file, std::string write_file) {
	char date[64];
	std::ofstream ofs(log_file, std::ios::app);   // 上書き
	time_t t = time(NULL);
	tm now;
	errno_t error;

	if (!ofs)
	{
		std::cout << "ファイルが開けませんでした。" << std::endl;
		std::cin.get();
		exit(9);
	}

	error = localtime_s(&now, &t);

	strftime(date, sizeof(date), "%Y/%m/%d %a %H:%M:%S\n", &now);
	ofs << date << "embedder::" << "read_file・" << read_file << "\n" << "write_file・" << write_file << "\n" << embed_file << "\n" << std::endl;
	ofs << "----------------------------------------------\n";
	std::cout << log_file << "に書き込みました。" << std::endl;

}


void str_checker(std::string read_file, std::string write_file) {
	// 本当は，連想配列を使うべきでは?   // 同じ部分は関数化すべき
	std::vector<std::string> r_label = { "Basket","Library","Lego", "Walk1", "Walk2" };
	std::vector<std::string> w_label = { "basket","library","lego", "walk1", "walk2" };

	std::vector<std::string> mv_label = { "basket","library","lego", "walk1", "walk2" };
	std::vector<std::string> m_array = { "m10", "m20", "m30" ,"m40" };
	std::vector<std::string> delta_array = { "d1" "d2", "d3" ,"d5", "d10" };


	if (write_file.find("test") != std::string::npos) {
		return;
	}

	// read_file , write_file check
	for (int i = 0; i < end(r_label) - begin(r_label); i++) {
		if (read_file.find(r_label[i]) != std::string::npos) {     // 当該文字列が含まれている
			if (write_file.find(w_label[i]) != std::string::npos) { // write_fileにも同じ文字列が含まれている
				break;
			}
			else {   // 同じ文字列が含まれていない．ミスってる場合
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

bool overwrite_check(std::string write_file) {       // うっかりデータを上書きしないようにするためのチェック関数
	std::string new_write_file = write_file + ".avi";
	std::ifstream ifs(new_write_file);

	if (ifs.is_open()) {
		std::cout << "error: overwrite::" << write_file << std::endl;
		getchar();
		return false;
	}

	return true;
}

void change_filename(std::string& read_file,  std::string& write_file, std::string& motion_vector_file, int loop_count) {
	const std::string mp4_read_array[5] = {"Basketball.avi", "Library.avi" , "Lego.avi", "Walk1.avi", "Walk2.avi"};
	const std::string read_array[5] = { "basket", "library","lego", "walk1", "walk2"};
	const std::vector<std::string> mv_label = { "basket","library","lego", "walk1", "walk2" };
	int change_point = 0;

	// read_fileの操作(loop_countに応じてファイルの名前の動画のタイトル部分を変更)
	change_point = (int)read_file.find("xxx");
	read_file.replace(change_point, 3, mp4_read_array[loop_count - 1]);

	// write_fileの操作
	change_point = (int)write_file.find("xxx");
	write_file.replace(change_point, 3, read_array[loop_count - 1]);

	
	// motion_vectorの操作
	change_point = (int)motion_vector_file.find("xxx");
	motion_vector_file.replace(change_point, 3, mv_label[loop_count - 1]);

	// コンソール出力
	std::cout << read_file  << std::endl;
	std::cout << write_file << std::endl;
}