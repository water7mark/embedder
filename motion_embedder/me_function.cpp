#include "me_header.h"

int delta_thisfile = 0;
float average_thisfile = 0;

static double cosine_table[block_height][block_width];  // DCT変換用のコサインテーブル

void init_me(cv::VideoCapture* cap, std::vector<char>* embed, cv::Size* size, std::ofstream* ofs, cv::VideoWriter* writer, std::string read_file, std::string write_file, int num_embedframe, std::vector<int>& inter_vec) {
	*embed = set_embeddata(embed_file);    
	set_interleave(interleave_file, inter_vec);
	*cap = capture_open(read_file);        
	//	*writer = mp4_writer_open(write_file + ".mp4", *cap);  // mp4なのでデータ量が小さいため分割の必要はない．．
	*writer = writer_open(write_file + "_1.avi", *cap);
	size->width = cap->get(CV_CAP_PROP_FRAME_WIDTH);
	size->height = cap->get(CV_CAP_PROP_FRAME_HEIGHT);
}

void set_ctable() {    //DCT変換で使うテーブルを初期設定
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


cv::Mat filter(cv::Mat luminance) {                           // ブロック内の輝度値をならす(輝度値の平均化)
	cv::Mat dst_luminance(luminance.size(), CV_32F);
	int x = 0, y = 0, m = 0, n = 0;
	for (m = 0; m < luminance.cols; m += block_width) {        //  blockごとに処理をする
		for (n = 0; n < luminance.rows; n += block_height) {
			float mean = 0, sum = 0;
			for (x = m; x < m + block_width; x++) {             // ブロック内の合計輝度値を求める
				for (y = n; y < n + block_height; y++) {
					sum += (float)luminance.at<uchar>(y, x);
				}
			}

			mean = sum / (block_width * block_height);        // 上記の合計輝度値の平均値
			for (x = m; x < m + block_width; x++) {
				for (y = n; y < n + block_height; y++) {
					dst_luminance.at<float>(y, x) = mean;
				}
			}
		}
	}
	return dst_luminance;
}

float median(std::vector<float> v) {     // 中央値を返す
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
//	// 連続する2フレームの輝度値から
//
//	// 要素がcv::Mat型の配列を指すポインタを受け取ってそれをもとに動き検出を行う
//	cv::Size size(FRAME_WIDTH, FRAME_HEIGHT);
//	cv::Mat lumi_diff(size, CV_8UC3);
//
//	for (int j = 0; j < FRAME_HEIGHT; j += block_height) {
//		for (int i = 0; i < FRAME_WIDTH; i += block_width) {
//			// ↓DAAD変換するとなぜかすべての画素の輝度値が大幅に上がるので，全ての画素の中で最も輝度値の変化が小さいものの変化量を足し合わせておきたい
//
//			for (int k = 0; k < block_height; k++) {
//				lumi_diff = std::abs(p_luminance.at<unsigned char>(j, i) - c_luminance.at<unsigned char>(j, i));  // 前後のフレームの輝度差を取る
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

void motion_embedder(std::vector<cv::Mat>& luminance, std::vector<cv::Mat> &dst_luminance,std::vector<char> embed, int cframe,int num_embedframe, int delta, std::vector<int>& inter_vec) {
	std::vector<cv::Mat> means;  //ブロック単位の平均輝度値を保持
	std::vector<cv::Mat> deviations;  //ブロック単位の平均値からの偏差を保持
	cv::Mat m_means = cv::Mat::zeros(1920, 1080, CV_32F);  //mフレーム間での「ブロック単位の平均値」の平均値を保持
	std::vector<cv::Mat> m_deviations;  //mフレーム間での「ブロック単位の平均値」の平均値からの偏差を保持
	int x, y;

	for (int i = 0; i < num_embedframe; i++) {    //  各画素の偏差とブロック内平均輝度値を求める
		cv::Mat temp_luminance = luminance[i].clone();
		means.push_back(filter(temp_luminance));    // luminanceの大きさでCV_32Fの配列が格納される
		temp_luminance.convertTo(temp_luminance, CV_32F);
		deviations.push_back(temp_luminance - means[i]);      // 各画素の偏差 = 各画素の輝度値-ブロック内平均輝度値 
	}

	//mフレームで平均をとる
	m_means = means[0].clone() / num_embedframe;  
	for (int i = 1; i < num_embedframe; i++) {
		m_means += means[i] / num_embedframe;
	}

	//mフレーム間での「ブロック単位の平均値」の平均値を保持
	std::vector<cv::Mat> t_variance(num_embedframe);
	cv::pow((means[0] - m_means), 2, t_variance[0]);
	cv::Mat variance = t_variance[0].clone();
	for (int i = 1; i < num_embedframe; i++) {
		cv::pow((means[i] - m_means), 2, t_variance[i]);
		variance += t_variance[i];
	}
	variance /= num_embedframe;





	/// 何か処理

	//埋め込み後フレームを返す
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


void motion_capture(cv::Mat cur_lumi, cv::Mat pre_lumi) {   // 2枚のフレームを取得して、動きのある部分を返す
	


	for (int y = 0; y < FRAME_HEIGHT; y++) {
		for (int x = 0; x < FRAME_WIDTH; x++) {
			if (pre_lumi.at<unsigned char>(y, x) == 255) {        // 白色なら
				if (cur_lumi.at<unsigned char>(y, x) == 255) {

				}
			}
		}
	}
	


}