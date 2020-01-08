// 前のプログラムはtemp01のリポジトリに入れてます


#include "me_header.h"

int delta_thisfile = 0;
float average_thisfile = 0;


#define dx 0
#define dy 1

static double cosine_table[block_height][block_width];  // DCT変換用のコサインテーブル
static int motionvector_block_size = 16; // 動きベクトルのグリッドブロックのサイズ

static int matrix_height = 128;
static int matrix_width = 120;

#define MV_DETA_SIZE CV_8SC1
#define NP_DETA_SIZE CV_16SC1

#define MV_DETA_TYPE char
#define NP_DETA_TYPE short


void init_me(cv::VideoCapture* cap, std::vector<char>* embed, cv::Size* size, std::ofstream* ofs, cv::VideoWriter* writer, std::string read_file, std::string write_file, std::string motion_vector_file, int num_embedframe) {
	*embed = set_embeddata(embed_file);
	*cap = capture_open(read_file);
	//	*writer = mp4_writer_open(write_file + ".mp4", *cap);  // mp4なのでデータ量が小さいため分割の必要はない．．
	*writer = mp4_writer_open(write_file + ".mp4", *cap);
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

void set_motionvector(const std::string motion_vector_file, std::vector<mv_class>& mv_all, int cframe) {
	// ファイル内のデータが膨大すぎるため、埋め込み時に適宜この関数を呼び出して、その都度ファイルからデータを読み出す
	// 埋め込む先頭フレーム(cframe)までファイル読み込みを飛ばしてからデータを取得する

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
	while (str.find(cframe_str) == std::string::npos) {   //pts=cframeとなるまで飛ばす
		getline(ifs, str, ' ');
	}


	// 書き換え必須
	for (int pts = 1; pts < num_embedframe && !ifs.eof();) {
		mv_class temp_class;
		int temp_start;
		std::string temp_str;
		int temp_count = 0;

		// 初期化
		temp_class.frame_index = -1;
		temp_class.x_vector = cv::Mat::zeros(cv::Size(1920, 1080), MV_DETA_SIZE);
		temp_class.y_vector = cv::Mat::zeros(cv::Size(1920, 1080), MV_DETA_SIZE);


		getline(ifs, str, ' ');

		if (str.find("pts") != std::string::npos) {          // ptsが出れば、1フレームとみなす 
			pts++;
		}
		else if (str.find("frame_index") != std::string::npos) {
			temp_start = str.find("frame_index");
			temp_str = str.substr(temp_start + 12, str.length());        //動きベクトルファイル内では、frame_index=数字になっているという前提
			temp_class.frame_index = atoi(temp_str.c_str());
		}
		else if (str.find("shape") != std::string::npos) {
			temp_start = str.find("shape");
			temp_str = str.substr(temp_start + 13, str.length());    // origin=video(若しくはdummy)となっている仮定 

			for (int i = 0; i < 120 * matrix_height; i++) {
				getline(ifs, str, '\t');

				if (i == 0) {
					temp_str = str.substr(13, 1);
				}
				else if (i % 120 == 0) {   // 行の切れ目の\nを削除
					temp_str = str.substr(1, 1);
				}
				else {
					temp_str = str.substr(0, 1);
				}

				if (i < 120 * matrix_height / 2) {
					temp_class.x_vector.at<MV_DETA_TYPE>(i / 120, i % 120) = atoi(temp_str.c_str());
				}
				else {
					temp_class.y_vector.at<MV_DETA_TYPE>(i / 120, i % 120) = atoi(temp_str.c_str());
				}
			}
		}

		mv_all.push_back(temp_class);
	}
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

void motion_embedder(std::vector<cv::Mat>& luminance, std::vector<cv::Mat> &dst_luminance, std::vector<char> embed, int cframe, int num_embedframe, int delta, std::string motion_vector_file, std::vector<mv_class>& mv_all) {
	std::vector<cv::Mat> means;  //ブロック単位の平均輝度値を保持
	std::vector<cv::Mat> deviations;  //ブロック単位の平均値からの偏差を保持
	cv::Mat m_means = cv::Mat::zeros(1920, 1080, CV_32F);  //mフレーム間での「ブロック単位の平均値」の平均値を保持
	std::vector<cv::Mat> m_deviations;  //mフレーム間での「ブロック単位の平均値」の平均値からの偏差を保持
	int x, y;

	// ファイルからデータ取得
	set_motionvector(motion_vector_file, mv_all, cframe);


	// means, m_means , deviations, variance の作成
	for (int i = 0; i < num_embedframe; i++) {    //  各画素の偏差とブロック内平均輝度値を求める
		cv::Mat temp_luminance = luminance[i].clone();
		means.push_back(filter(temp_luminance));
		temp_luminance.convertTo(temp_luminance, CV_32F);
		deviations.push_back(temp_luminance - means[i]);      // 各画素の偏差 = 各画素の輝度値-ブロック内平均輝度値 
	}

	//mフレーム間での「ブロック単位の平均値」の平均値を保持
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
	// 作成終了



	//std::vector<next_pos_all> frame_pos_all;  // 20フレーム分の座標の次の移動座標を格納している
	cv::Mat Is_stop = cv::Mat::zeros(cv::Size(1920, 1080), CV_8UC1);     // 20フレームの間でその画素が他の画素によって上書きされるなら1を入れる。そうでなければ0のまま
	cv::Mat Is_move = cv::Mat::zeros(cv::Size(1920, 1080), CV_8UC1);   // 20フレームのどこかで移動する場合はtrueになる、一度も移動しないならfalse


	//next_pos_all temp_class;
	//next_pos_all temp_class2;

	std::vector<cv::Mat> result_lumi;
	std::vector<float> lumi(20, 0);         //　初期化必要
	std::vector<cv::Mat> lumi_map;

	// 先に動きベクトルの処理
	for (y = 0; y < FRAME_HEIGHT; y += block_height) {
		for (x = 0; x < FRAME_WIDTH; x += block_width) {
			// 先に20フレーム間で当該画素が移動するかを調べる
			for (int i = 0; i < num_embedframe; i++) {

				//temp_class.now_y = cv::Mat::zeros(cv::Size(1920, 1080), NP_DETA_SIZE);
				//temp_class.now_x = cv::Mat::zeros(cv::Size(1920, 1080), NP_DETA_SIZE);

				if (i == 0) {      // i=0は自身を先頭に格納	
					//temp_class2.now_y = cv::Mat::zeros(cv::Size(1920, 1080), NP_DETA_SIZE);
					//temp_class2.now_x = cv::Mat::zeros(cv::Size(1920, 1080), NP_DETA_SIZE);
					//temp_class2.now_y.at<NP_DETA_TYPE>(y,x) = y;
					//temp_class2.now_x.at<NP_DETA_TYPE>(y,x) = x;
					//frame_pos_all.push_back(temp_class2);

					lumi_map[i].at<float>(y, x) = means[i].at<float>(y, x);
				}
				else if (i == num_embedframe - 1) {
					continue;
				}


				if (Is_there_mv(mv_all, cframe + i)) {  // 現在のフレーム番号を与えると動きベクトルが出力されているか返す関数					
					std::pair<int, int> next_pixel = get_next_pos(mv_all, cframe + i, y, x);
					//temp_class.now_y.at<NP_DETA_TYPE>(y,x) = next_pixel.first;
					//temp_class.now_x.at<NP_DETA_TYPE>(y,x) = next_pixel.second;
					//frame_pos_all.push_back(temp_class);

					lumi_map[i + 1].at<float>(y, x) = means[i + 1].at<float>(next_pixel.first, next_pixel.second);

					Is_stop.at<unsigned char>(next_pixel.first, next_pixel.second) = 1;
					Is_move.at<unsigned char>(y, x) = 1;
				}
				else {       // 今と変わらない場合は、現在の座標を格納する
					//temp_class.now_y.at<NP_DETA_TYPE>(y, x) = y;
					//temp_class.now_x.at<NP_DETA_TYPE>(y, x) = x;
					//frame_pos_all.push_back(temp_class);

					lumi[i + 1] = means[i + 1].at<float>(y, x);
				}
			}
		}
	}


	//frame_pos_allに3種類(動かないもの、動いているもの、上書きされるものの3つが格納できているか考える必要がある。)
	// 一応上まではできた。以下のプログラムがまだ、

	float ave_lumi;
	float var_lumi;
	float num;
	
	

	// 埋め込み処理
	for (y = 0; y < FRAME_HEIGHT; y += block_height) {
		for (x = 0; x < FRAME_WIDTH; x += block_width) {
			if (Is_stop.at<unsigned char>(y, x) = true) {      // ここで上書きされているかどうかを調べる、上書きされているなら埋め込みはしない
				continue;
			}
			else if (Is_move.at<unsigned char>(y, x) == 1) {
				num = (embed[(x / block_width) % BG_width + ((y / block_height) % BG_height)*BG_width] == '0') ? 0 : 1;

				//動いた画素の20フレーム分の平均と分散を求める
				ave_lumi = 0;
				var_lumi = 0;

				for (int i = 0; i < num_embedframe; i++) {        // 当該画素における20フレーム分の輝度を集める
					lumi[i] = lumi_map[i].at<float>(y, x);
					ave_lumi += lumi[i];
				}
				ave_lumi /= 20;
				for (int i = 0; i < num_embedframe; i++) {
					var_lumi = pow((lumi[i] - ave_lumi), 2);
				}
				// 平均と分散求め終わり


				if (num == 0) {
					for (int t_delta = delta; t_delta >= 1; t_delta--) {   // deltaよりもどの程度分散が大きいかどうかで操作する量を決めている
						if (var_lumi >= t_delta * t_delta) {
							operate_lumi_for_zero(lumi, ave_lumi, var_lumi, t_delta);
						}
					}

					for (int i = 0; i < num_embedframe; i++) {              // 操作した分を反映させる
						result_lumi[i].at<float>(frame_pos_all[i].now_y.at<NP_DETA_TYPE>(y, x), frame_pos_all[i].now_x.at<NP_DETA_TYPE>(y, x)) = lumi[i];
					}
				}
				else {
					for (int i = 0; i < num_embedframe; i++) {
						operate_lumi_for_one(lumi, ave_lumi, var_lumi, delta);
						result_lumi[i].at<float>(y, x) = lumi[i];
					}
				}
			}

			else {                                                 // 従来通り
				float num = (embed[(x / block_width) % BG_width + ((y / block_height) % BG_height)*BG_width] == '0') ? 0 : 1;
				var_lumi = variance.at<float>(y, x);

				if (num == 0) {  // 透かしビットが0の時
					for (int i = 0; i < num_embedframe; i++) {
						lumi[i] = means[i].at<float>(y, x);
					}

					for (int t_delta = delta; t_delta >= 1; t_delta--) {   // deltaよりもどの程度分散が大きいかどうかで操作する量を決めている
						if (var_lumi >= t_delta * t_delta) {
							operate_lumi_for_zero(lumi, m_means.at<float>(y, x), var_lumi, t_delta);
						}
					}

					for (int i = 0; i < num_embedframe; i++) {
						result_lumi[i].at<float>(y, x) = lumi[i];
						lumi[i] = 0;
					}
				}
				else {    // 透かしビットが1の時
					for (int i = 0; i < num_embedframe; i++) {
						operate_lumi_for_one(lumi, m_means.at<float>(y, x), var_lumi, delta);
						result_lumi[i].at<float>(y, x) = lumi[i];
					}
				}
			}
		}
	}

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

void operate_lumi_for_one(std::vector<float> &lumi, float average, float variance, int delta) {
	size_t index_max, index_min; // 最大、最小の要素の添え字
	size_t num_low_ave = 0;  // 平均よりも低い個数
	size_t num_high_ave = 0; //平均よりも高い個数
	float now_variance;

	int p_cnt = 0;
	int m_cnt = 0;

	for (int i = 0; i < num_embedframe; i++) {
		if ((lumi[i] >= average && p_cnt != num_embedframe / 2) || m_cnt == num_embedframe / 2) {
			lumi[i] += delta;  //δ加えたフレームの数
			p_cnt++;
		}
		else {
			lumi[i] -= delta;  //-δ加えたフレームの数
			m_cnt++;
		}
	}
}

void operate_lumi_for_zero(std::vector<float> &lumi, float average, float variance, int delta) {  // 平均を維持しつつ、標準偏差を分散未満にする関数
	// average, varianceはlumiで与えられる輝度値の平均と分散であり、deltaは埋め込み強度
	size_t index_max, index_min; // 最大、最小の要素の添え字
	size_t num_low_ave = 0;  // 平均よりも低い個数
	size_t num_high_ave = 0; //平均よりも高い個数
	float now_variance;
	std::vector<double> temp_lumi(20);

	average_thisfile = average;


	// 計算用の配列に格納
	for (int i = 0; i < end(lumi) - begin(lumi); i++) {
		temp_lumi[i] = lumi[i];
	}


	for (int limit_time = 0; limit_time < 30; limit_time++) {
		// 平均から最も遠い要素のインデックスを求める
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


	// 元に戻す
	for (int i = 0; i < end(lumi) - begin(lumi); i++) {
		lumi[i] = temp_lumi[i];
	}
}



int ptob(int pixel_pos) {
	return pixel_pos / motionvector_block_size;
}

int btop(int block_pos) {
	return block_pos * motionvector_block_size;
}

std::pair<int, int > get_next_pos(std::vector<mv_class>& mv_all, int frame, int y, int x) {
	// フレーム番号と座標を与えるとその次の座標を返す)
	// 何も動いていなければ元の座標を返す
	int bl_y = ptob(y);
	int bl_x = ptob(x);

	std::pair<int, int> next_pos = std::make_pair(btop(mv_all[frame].y_vector.at<MV_DETA_TYPE>(bl_y, bl_x)) + y, btop(mv_all[frame].x_vector.at<MV_DETA_TYPE>(bl_y, bl_x)) + x);

	return next_pos;
}

bool Is_there_mv(std::vector<mv_class> &mv_all, int frame) {  // ダミー場合はfalseを返す

	if (mv_all[frame].frame_index == -1) {
		return false;
	}
	else {
		return true;
	}
}