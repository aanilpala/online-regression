package regression.offline.learner;

import regression.online.model.Prediction;

public class KernelRegressionBatch extends BatchRegressor {

	double[] h; // bandwidths;
	double[] density_estimates;
	double[][][] training_set;
	double[][] responses;
		
	public KernelRegressionBatch(int input_width, int training_set_size) {
		
		super(input_width, training_set_size, false);
		
		h = new double[input_width];
		density_estimates = new double[training_set_size];
		training_set = new double[training_set_size][input_width][1];
		responses = new double[training_set_size][1];
	}	
	
	@Override
	public double predict(double[][] org_dp) throws Exception {
		
		double[][] dp = scale_input(org_dp);
		
		double pp = 0;
		double denom = 0;
		
		for(int ctr = 0; ctr < training_set_size; ctr++) {
			double kernel_measure = kernel_func(training_set[ctr], dp, h);
			pp += responses[ctr][0]*kernel_measure;
			denom += kernel_measure;
		}
		
		if(denom == 0) return 0.0;
		
		pp /= denom;
		
		return target_postscaler(pp);
	}
	
	@Override
	public void train(double[][][] dps, double[][] targets) throws Exception {
		compute_means(dps, targets);
		determine_scaling_factors();
		scale_training_data(dps, targets);
		
		for(int ctr = 0; ctr < training_set_size; ctr++) {
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				training_set[ctr][ctr2][0] = dps[ctr][ctr2][0];
			}
		}
		
		for(int ctr = 0; ctr < training_set_size; ctr++) {
			responses[ctr][0] = targets[ctr][0];
		}
		
		//tuning
		
		double[][] weights_cov = get_weights_cov_matrix(training_set);
		
		double step_size_factor = 0.05;
		double end_factor = 2;
		double start_factor = 0.05;
		double opt_factor = -1;
		
		double target_rss = Double.POSITIVE_INFINITY; // initialization
		
		for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
			h[ctr2] = 0;
		}
		
		for(double cur_factor = start_factor; cur_factor < end_factor; cur_factor += step_size_factor) {
			
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				h[ctr2] += weights_cov[ctr2][ctr2]*step_size_factor;
			}
			
			double exp_rss = get_hold_one_out_rss(h);
			
//			System.out.println(exp_rss);
			
			if(exp_rss < target_rss) {
				target_rss = exp_rss;
				opt_factor = cur_factor;
			}
			
		}
		
//		System.out.println(opt_factor);
		
		if(opt_factor != -1.0) {
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				h[ctr2] = opt_factor*weights_cov[ctr2][ctr2];
			}
		}
		
		
	}
	
	private double get_hold_one_out_rss(double[] experimental_bandwidth_array) {
		
		long rss = 0;
		
		for(int ctr = 0; ctr < training_set_size; ctr++) {
			
			double density = 0;
			double pp = 0;
			
			for(int ctr2 = 0; ctr2 < training_set_size; ctr2++) {
				if(ctr == ctr2) continue;
				
				double kernel_measure = kernel_func(training_set[ctr], training_set[ctr2], experimental_bandwidth_array);
				
				pp += kernel_measure*responses[ctr2][0];
				density += kernel_measure;
				
			}
			
			if(density == 0) return Double.POSITIVE_INFINITY;  // we strictly don't want NaNs in the predictions!
			
			pp /= density;
			
			double residual = (pp - responses[ctr][0])*1000;
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
