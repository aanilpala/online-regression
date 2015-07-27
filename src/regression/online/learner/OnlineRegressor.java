package regression.online.learner;
import regression.online.model.Prediction;
import regression.util.NLInputMapper;


public abstract class OnlineRegressor {

	public int feature_count;
	protected NLInputMapper nlinmap;
	public boolean map2fs;
	public String name;
	
	public int update_count;
	public int high_error_start_point;
	public int burn_in_count;
	
	protected double[] input_scaler;
	protected double target_scaler;
	protected double[] input_means;
	protected double target_mean;
	
	double m_new, m_old, s_new, s_old;
	
	protected boolean verbouse = false;
	
	public OnlineRegressor(boolean map2fs, int input_width) {
		
		this.map2fs = map2fs;
		
		name = this.getClass().getName() + (map2fs ? "Mapped" : "");
		
		if(map2fs) {
			nlinmap = new NLInputMapper(input_width, 2, true, true);
			this.feature_count = nlinmap.feature_dim;
		}
		else {
			this.feature_count = input_width;
		}
		
		update_count = 0;
		high_error_start_point = 0;
		
		input_scaler = new double[feature_count];
		input_means = new double[feature_count];
		
		for (int ctr = 0; ctr < input_scaler.length; ctr++)
			input_scaler[ctr] = 1;
		
		target_scaler = 1;
		
		for (int ctr = 0; ctr < input_means.length; ctr++)
			input_means[ctr] = 0;
		
		target_mean = 0;
		
		m_old = 0; m_new = 0;
		s_old = 0; s_new = 0;
	}
	
	public void update_running_means(double[][] dp, double target) {
		
		target_mean = (target_mean*((update_count - high_error_start_point)-1) + Math.abs(target))/(update_count - high_error_start_point);
		
		for (int ctr = 0; ctr < input_means.length; ctr++)
			input_means[ctr] = (input_means[ctr]*((update_count - high_error_start_point)-1) + dp[ctr][0])/(update_count - high_error_start_point);
		
	}
	
	public double[][] scale_input(double[][] dp) {
		
		double[][] scaled_dp = new double[dp.length][1];
		
		for (int ctr = 0; ctr < input_scaler.length; ctr++)
			scaled_dp[ctr][0] = input_scaler[ctr]*dp[ctr][0];
		
		return scaled_dp;
	}
	
	public double[][] revert_input(double[][] scaled_dp) {
		
		double[][] dp = new double[scaled_dp.length][1];
		
		for (int ctr = 0; ctr < input_scaler.length; ctr++)
			dp[ctr][0] = (1/input_scaler[ctr])*scaled_dp[ctr][0];
		
		return dp;
	}
	
	public double target_prescaler(double target) {
		return target_scaler*target;
	}
	
	public double target_postscaler(double target) {
		return (1.0/target_scaler)*target;
	}
	
	public Prediction predict(double[][] dp) throws Exception { return null; };
	public boolean update(double[][] dp, double y, Prediction prediction) throws Exception { return false; };

	public int get_burn_in_number() {return burn_in_count; };
	
	public String get_name() {
		return name.split("\\.")[3];  
	}

	public boolean is_tuning_time() { return false;}

	public boolean was_just_tuned() { return false;}
	
	public void update_scaling_factors() {
		
		if(target_mean != 0)
			target_scaler = 1/(target_mean);
		
		for (int ctr = 0; ctr < input_means.length; ctr++) {
			if(input_means[ctr] == 0) continue;
			input_scaler[ctr] = 1/(input_means[ctr]);
		}
	}
	
}
