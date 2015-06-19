package regression.online.learner;

import java.util.Arrays;

import regression.online.model.Prediction;

public class KernelRegression extends WindowRegressor {

	double[] h; // bandwidths;
	double[] density_estimates;
	double[] contributions;
	
	double init_bandwidth = 1;
	int hyper_param_tuning_freq = 100*w_size;
	
	public KernelRegression(int input_width, int window_size) {
		
		super(false, input_width, window_size);
		
		h = new double[input_width];
		density_estimates = new double[w_size];
		contributions = new double[w_size];
		
		for(int ctr = 0; ctr < input_width; ctr++)
			h[ctr] = init_bandwidth;
	}
	
	@Override
	public Prediction predict(double[][] dp) throws Exception {
		
		if(!slide && w_start == w_end) return new Prediction();
		
		double pp = 0;
		double denom = 0;
		
		for(int ptr = w_start, ctr = 0; ctr < n; ptr = (ptr + 1) % w_size, ctr++) {
			double kernel_measure = kernel_func(dp_window[ptr], dp, h);
			pp += responses[ptr][0]*kernel_measure;
			denom += kernel_measure;
		}
		
		if(denom == 0) return new Prediction();
		
		pp /= denom;
		
		// calculating predictive density
		
		long temp = 0;
		
		for(int ptr = w_start, ctr = 0; ctr < n; ptr = (ptr + 1) % w_size, ctr++) {
			long residual;
		
			if(density_estimates[ptr] == 0) residual = (long) (responses[ptr][0]*10000);
			else residual = (long) ((responses[ptr][0] - (contributions[ptr]/density_estimates[ptr]))*10000);
			
			temp += residual*residual;
		}
				
		double sigma_square = (double) temp/(100000000.0*n);
		
//		if(slide) {
//			System.out.println(sigma_square);
//		}
		
		double predictive_deviance = Math.sqrt(((Math.pow(4*Math.PI, -0.5*feature_count))*sigma_square)/(denom));
		
		return new Prediction(pp, predictive_deviance);
	}
	
	@Override
	public void update(double[][] dp, double y, Prediction prediction) throws Exception {
		
		int index = getIndexForDp(dp);
		
		if(index != -1) {
			// reject the update
			// avg the response for the duplicate point
			
			responses[index][0] = (y + responses[index][0])/2.0;
			return;
		} 
		
		if(slide) {
			for(int ptr = w_start, ctr = 0; ctr < n-1; ptr = (ptr + 1) % w_size, ctr++) {
				double kernel_measure = kernel_func(dp_window[ptr], dp_window[w_end], h);
				density_estimates[ptr] -= kernel_measure;
				contributions[ptr] -= kernel_measure*responses[w_end][0];
			}
		}
		
		dp_window[w_end] = Arrays.copyOfRange(dp, 0, feature_count);
		responses[w_end][0] = y;
		density_estimates[w_end] = 0;
		contributions[w_end] = 0;
		
		for(int ptr = w_start, ctr = 0; ctr < n-1; ptr = (ptr + 1) % w_size, ctr++) {
			double kernel_measure = kernel_func(dp_window[ptr], dp, h);
			density_estimates[w_end] += kernel_measure;
			contributions[w_end] += kernel_measure*responses[ptr][0];
			
			density_estimates[ptr] += kernel_measure;
			contributions[ptr] += kernel_measure*y;
		}
		
		double kernel_measure = kernel_func(dp, dp, h);
		density_estimates[w_end] += kernel_measure;
		contributions[w_end] += kernel_measure*y;
		
		if(slide) {
			w_start = (w_start + 1) % w_size;
			w_end = (w_end + 1) % w_size;
		}
		else {
			w_end = (w_end + 1) % w_size;
			if(w_end == w_start) slide = true;
		}
		
		count_dps_in_window();
		
		update_count++;
		if((update_count - w_size) % hyper_param_tuning_freq == 0) {
			tune_hyper_params();
			recompute_past_kernel_densities();
		}
		
	};
	
	private void recompute_past_kernel_densities() {
		
		for(int ptr = w_start, ctr = 0; ctr < n; ptr = (ptr + 1) % w_size, ctr++) {
			
			double density = 0;
			double contributions = 0;
			
			for(int ptr2 = w_start, ctr2 = 0; ctr2 < n; ptr2 = (ptr2 + 1) % w_size, ctr2++) {
				//if(ptr == ptr2) continue;
				
				double kernel_measure = kernel_func(dp_window[ptr], dp_window[ptr2], h);
				
				density += kernel_measure;
				contributions += kernel_measure*responses[ptr2][0];
				
			}
			
			density_estimates[ptr] = density;
			this.contributions[ptr] = contributions;
		}
		
	}

	private void tune_hyper_params() throws Exception {
		
		// get rss with the current bandwidths
		double target_rss = get_hold_one_out_rss(h);
		//System.out.println(target_rss);

		// get the variance
		double[][] weights_cov = get_weights_cov_matrix();
		
		double[] experimental_bandwidth_array = new double[feature_count];
		
		// init
		for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
			experimental_bandwidth_array[ctr2] = 0;
		}
		
		double step_size_factor = 0.001;
		double end_factor = 0.1;
		double start_factor = 0.001;
		double opt_factor = -1;
		
		for(double cur_factor = start_factor; cur_factor < end_factor; cur_factor += step_size_factor) {
			
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				experimental_bandwidth_array[ctr2] += weights_cov[ctr2][ctr2]*step_size_factor;
			}
			
			double exp_rss = get_hold_one_out_rss(experimental_bandwidth_array);
			
//			System.out.println(exp_rss);
			
			if(exp_rss < target_rss) {
				target_rss = exp_rss;
				opt_factor = cur_factor;
			}
			
		}
		
		if(opt_factor != -1.0) {
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				h[ctr2] = opt_factor*weights_cov[ctr2][ctr2];
			}
		}
		
	}
	
	private double get_hold_one_out_rss(double[] experimental_bandwidth_array) {
		
		long rss = 0;
		
		for(int ptr = w_start, ctr = 0; ctr < n; ptr = (ptr + 1) % w_size, ctr++) {
			
			double density = 0;
			double pp = 0;
			
			for(int ptr2 = w_start, ctr2 = 0; ctr2 < n; ptr2 = (ptr2 + 1) % w_size, ctr2++) {
				if(ptr == ptr2) continue;
				
				double kernel_measure = kernel_func(dp_window[ptr], dp_window[ptr2], experimental_bandwidth_array);
				
				pp += kernel_measure*responses[ptr2][0];
				density += kernel_measure;
				
			}
			
			if(density == 0) return Double.POSITIVE_INFINITY;  // we strictly don't want NaNs in the predictions!
			
			pp /= density;
			
			double residual = (pp - responses[ptr][0])*1000;
			rss += residual*residual;
		}
		
		return Math.sqrt(rss/1000000.0);
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
	
	

}