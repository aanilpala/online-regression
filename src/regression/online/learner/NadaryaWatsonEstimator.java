package regression.online.learner;

import regression.online.model.Prediction;

public class NadaryaWatsonEstimator extends WindowRegressor {
	
	double[] h; // bandwidths
	double[] var; // variance of features
	
	double init_bandwidth = 10;
	
	int hyper_param_tuning_freq = 1000;
	
	int update_count;
	
	double[][] fitted;
	
	public NadaryaWatsonEstimator(int input_width) {
		
		super(false, input_width);
		
		name = "Nadarya-Watson";
		
		fitted = new double[w_size][2];
		h = new double[input_width];
		var = new double[input_width];
		
		for(int ctr = 0; ctr < input_width; ctr++)
			h[ctr] = init_bandwidth;
		
		update_count = 0;
		
	}

	private static double kernel_func(double[][] dp1, double dp2[][], double[] h) {
		
		int dim = dp1.length;
		long mult = 1;
		
		for(int ctr = 0; ctr < dim; ctr++) {
			double scaled_dif = (dp1[ctr][0] - dp2[ctr][0])/h[ctr];
			mult *= Math.pow(2*Math.PI, -0.5)*Math.pow(Math.E, -0.5*scaled_dif*scaled_dif)*1000;
		}
		
		double kernel_measure = mult / Math.pow(1000, dim); 
		
		return kernel_measure;
	}
	
	
	@Override
	public Prediction predict(double[][] dp) {
		
		
		if(!slide && w_start == w_end) return new Prediction();
		
		count_dps_in_window(); // this sets n;
		
		double denom = 0;
		double nom = 0;
		
		for(int ptr = w_start, ctr = 0; ctr < n; ptr = (ptr + 1) % w_size, ctr++) {
			double kernel_measure = kernel_func(dp_window[ptr], dp, h);
			nom += kernel_measure*responses[ptr][0];
			denom += kernel_measure;
		}
		
		double pp = nom/denom;
		
		long temp = 0;
		
		for(int ptr = w_start, ctr = 0; ctr < n; ptr = (ptr + 1) % w_size, ctr++) {
			long residual;
		
			if(fitted[ptr][0] == 0 && fitted[ptr][1] == 0) residual = (long) (responses[ptr][0]*10000);
			else residual = (long) ((responses[ptr][0] - (fitted[ptr][0]/fitted[ptr][1]))*10000);
			
			//System.out.println(fitted[ptr][0] + "-" + fitted[ptr][1]/fitted[ptr][2] + "=" + fitted[ptr][1] + "/" + fitted[ptr][2]);
			
			temp += residual*residual;
		}
				
		double sigma_square = (double) temp/(100000000.0*n);
		
		
		double predictive_deviation = Math.sqrt(((Math.pow(4*Math.PI, -0.5*feature_count))*sigma_square)/(denom));
		
		return new Prediction(pp, pp + 1.96*predictive_deviation, pp - 1.96*predictive_deviation, nom, denom);
	}

	@Override
	public void update(double[][] dp, double y, Prediction prediction) {
		
		count_dps_in_window(); // this sets n;
		
		if(slide) {
			
			// window is full
			// replace the element w_start points with the new element
			// increment w_start and w_end by 1
			
			double[][] to_remove = dp_window[w_start];
			
			for(int ptr = w_start; ptr != w_end; ptr = (ptr + 1) % w_size) {
				double kernel_measure = kernel_func(to_remove, dp_window[ptr], h); 
				fitted[ptr][1] -= kernel_measure;
				fitted[ptr][0] -= kernel_measure*responses[w_start][0];
			}
			
			for(int ctr = 0; ctr < feature_count; ctr++) {
				dp_window[w_end][ctr][0] = dp[ctr][0];
			}
			
			for(int ptr = w_start, ctr = 0; ctr < n; ptr = (ptr + 1) % w_size, ctr++) {
				double kernel_measure = kernel_func(dp_window[w_end], dp_window[ptr], h); 
				fitted[ptr][1] += kernel_measure;
				fitted[ptr][0] += kernel_measure*y;
			}
						
			// adding the effect of itself on the estimation! (next time for an identical dp, this data point will be utilized so residuals should be computed taking this fact into account)
			double kernel_measure = kernel_func(dp_window[w_end], dp_window[w_end], h); 
			
			responses[w_end][0] = y;
			fitted[w_end][0] = prediction.opt1 + kernel_measure*y;
			fitted[w_end][1] = prediction.opt2 + kernel_measure;
			
			w_start = (w_start + 1) % w_size;
			w_end = (w_end + 1) % w_size;
			
		}
		else {
			
			for(int ctr = 0; ctr < feature_count; ctr++) {
				dp_window[w_end][ctr][0] = dp[ctr][0];
			}
			
			for(int ptr = w_start, ctr = 0; ctr < n; ptr = (ptr + 1) % w_size, ctr++) {
				double kernel_measure = kernel_func(dp_window[w_end], dp_window[ptr], h); 
				fitted[ptr][1] += kernel_measure;
				fitted[ptr][0] += kernel_measure*y;
			}
						
			// adding the effect of itself on the estimation! (next time for an identical dp, this data point will be utilized so residuals should be computed taking this fact into account)
			double kernel_measure = kernel_func(dp_window[w_end], dp_window[w_end], h); 
			
			responses[w_end][0] = y;
			fitted[w_end][0] = prediction.opt1 + kernel_measure*y;
			fitted[w_end][1] = prediction.opt2 + kernel_measure;
			
			w_end = (w_end + 1) % w_size;
			
			if(w_end == w_start) slide = true;
			
		}
		
		update_count++;
		if((update_count - w_size) % hyper_param_tuning_freq == 0)
			tune_hyper_params();
		
	}

	private void tune_hyper_params() {

		count_dps_in_window(); // this sets n;
		
		// get rss with the current bandwidths
		double target_rss = get_hold_one_out_rss(h);
		
		//System.out.println(target_rss);

		// get the variance
		double[] var = new double[feature_count];
		double[] mean = new double[feature_count];
		
		for(int ptr = w_start, ctr = 0; ctr < n; ptr = (ptr + 1) % w_size, ctr++) {
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				mean[ctr2] += dp_window[ptr][ctr2][0];
			}
		}

		for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
			mean[ctr2] /= n;
		}
		
		for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
			var[ctr2] = 0;
		}
		
		for(int ptr = w_start, ctr = 0; ctr < n; ptr = (ptr + 1) % w_size, ctr++) {
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				var[ctr2] += Math.pow((dp_window[ptr][ctr2][0] - mean[ctr2]), 2);
			}
		}
		
		for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
			var[ctr2] = var[ctr2]/n;
		}
		
		double[] experimental_bandwidth_array = new double[feature_count];
		double start_factor = 0.1;
		double end_factor = 2.0;
		double step_size_factor = 0.1;
		double best_factor = -1.0;
		
		// init
		for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
			experimental_bandwidth_array[ctr2] = 0;
		}
		
		for(double cur_factor = start_factor; cur_factor < end_factor; cur_factor += step_size_factor) {
			
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				experimental_bandwidth_array[ctr2] += var[ctr2]*step_size_factor;
			}
			
			double exp_rss = get_hold_one_out_rss(experimental_bandwidth_array);
			
			if(exp_rss < target_rss) best_factor = cur_factor; 
			
		}
		
		if(best_factor != -1.0) {
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				h[ctr2] = best_factor*var[ctr2];
			}
		}
		
	}
	
	private double get_hold_one_out_rss(double[] experimental_bandwidth_array) {
		
		long rss = 0;
		
		for(int ptr = w_start, ctr = 0; ctr < n; ptr = (ptr + 1) % w_size, ctr++) {
			
			double denom = 0;
			double nom = 0;
			
			for(int ptr2 = w_start, ctr2 = 0; ctr2 < n; ptr2 = (ptr2 + 1) % w_size, ctr2++) {
				if(ptr == ptr2) continue;
				
				double kernel_measure = kernel_func(dp_window[ptr], dp_window[ptr2], experimental_bandwidth_array);
				
				nom += kernel_measure*responses[ptr2][0];
				denom += kernel_measure;
				
			}
			
			double residual = (nom/denom - responses[ptr][0])*1000;
			rss += residual*residual;
		}
		
		return Math.sqrt(rss/1000000.0);
	}
	
}
