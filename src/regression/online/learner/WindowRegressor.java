package regression.online.learner;

public abstract class WindowRegressor extends Regressor {

	public int w_size = 40; // sliding window size
	public int w_start, w_end;
	public int n; // number of data points in the window
	double[][][] dp_window;
	double[][] responses;
	boolean slide;
	int update_count;
	
	public void count_dps_in_window() {
		if(slide) n = w_size;
		else n = w_end - w_start;
	}
	
	public WindowRegressor(boolean map2fs, int input_width) {
		
		super(map2fs, input_width);
		
		dp_window = new double[w_size][feature_count][1];
		responses = new double[w_size][1];
		
		slide = false;
		w_start = 0;
		w_end = 0;
		update_count = 0;
	}
	
	
	
	
	
	
}
