package regression.online.learner;
import regression.online.model.Prediction;
import regression.online.util.NLInputMapper;


public abstract class Regressor {

	public int feature_count;
	NLInputMapper nlinmap;
	public boolean map2fs;
	public String name;
	
	double[] input_scaler;
	double target_scaler;
	
	int update_count;
	int burn_in_count;
	
	double target_mean;
	double[] input_means;
	
	private long running_squared_error;
	protected int prediction_count;
	double m_new, m_old, s_new, s_old;
	double smse_limit = 0.25;
	double smse_limit_raiser = 1.1;
	
	boolean update_inhibator;
	
	boolean verbouse = false;
	
	public Regressor(boolean map2fs2, int input_width, boolean update_inhibator) {
		
		this.update_inhibator = update_inhibator;
		
		this.map2fs = map2fs2;
		
		name = this.getClass().getName() + (update_inhibator ? "BATCH" : "");
		
		if(update_inhibator)
			name.replace("WINDOWED", "");
		
		if(map2fs) {
			nlinmap = new NLInputMapper(input_width, 2, true, true);
			this.feature_count = nlinmap.feature_dim;
		}
		else {
			this.feature_count = input_width;
		}
		
		update_count = 0;
		
		input_scaler = new double[feature_count];
		input_means = new double[feature_count];
		
		for (int ctr = 0; ctr < input_scaler.length; ctr++)
			input_scaler[ctr] = 1;
		
		target_scaler = 1;
		
		for (int ctr = 0; ctr < input_means.length; ctr++)
			input_means[ctr] = 0;
		
		target_mean = 0;
		
		running_squared_error = 0;
		prediction_count = 0;
		m_old = 0; m_new = 0;
		s_old = 0; s_new = 0;
	}
	
	public void update_running_means(double[][] dp, double target) {
		
		target_mean = target_mean + (target - target_mean)/update_count;
		
		for (int ctr = 0; ctr < input_means.length; ctr++)
			input_means[ctr] = input_means[ctr] + (dp[ctr][0] - input_means[ctr])/update_count;
		
	}
	
	public boolean high_error() {
		
		if(prediction_count < burn_in_count/2) return false;
		
		double target_variance = s_new/(float) (prediction_count-1);
		double residual_variance = (running_squared_error / (long) prediction_count) / 10000.0;
		double smse = residual_variance/target_variance;
		
		if(smse > smse_limit) {
			
			smse_limit *= smse_limit_raiser;
			//System.out.println(prediction_count + "-" + smse);
			
			running_squared_error = 0;
			prediction_count = 0;
			m_old = 0; m_new = 0;
			s_old = 0; s_new = 0;
			return true;
		}
		else return false;
	}
	
	public void update_prediction_errors(double target, double pp) {
		prediction_count++;
		if(prediction_count == 1) {
			m_old = m_new = target;
			s_old = 0;
		}
		else {
			m_new = m_old + (target - m_old)/((float) prediction_count);
			s_new = s_old + (target - m_old)*(target - m_new);
			
			//for the next iteration
			m_old = m_new;
			s_old = s_new;
		}
		//err = Math.abs(pred.point_prediction - response);
		running_squared_error += Math.pow((pp - target)*100, 2);
	}
	
	
	public void update_scaling_factors() {
		
		if(target_mean != 0)
			target_scaler = 1/(target_mean);
		
		for (int ctr = 0; ctr < input_means.length; ctr++) {
			if(input_means[ctr] == 0) continue;
			input_scaler[ctr] = 1/(input_means[ctr]);
		}
		
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
	public void update(double[][] dp, double y, Prediction prediction) throws Exception {}

	public int get_burn_in_number() {return burn_in_count; };
	
	public String get_name() {
		return name.split("\\.")[3];  
	}

	public boolean is_tuning_time() { return false;}

	public boolean was_just_tuned() { return false;}
	
	
}
