package regression.online.learner;

public abstract class WindowRegressor extends Regressor {

	public int w_size = 250; // sliding window size
	public int w_start, w_end;
	public int n; // number of data points in the window
	double[][][] dp_window;
	double responses[];
	
	public void count_dps_in_window() {
		if((w_end + 1) % w_size == w_start) n = w_size - 1;
		else n = w_end - w_start;
	}
	
	public WindowRegressor(boolean map2fs, int input_width) {
		
		super(map2fs, input_width);
		
		dp_window = new double[w_size][feature_count][1];
		responses = new double[w_size];
	}
	
	
	
	
	
	
}
