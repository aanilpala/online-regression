package regression.online.learner;

import java.io.IOException;
import java.util.Random;

import regression.online.util.MatrixOp;

public abstract class GPWindowedBase extends WindowRegressor{

	double[][] k; // gram matrix
	double[][] k_inv; // inverse gram matrix	
	
	// hyperparams
	double a; //measurement_precision;
	double b; //weight_precision;
	double[] hyperparams; // length-scale array;
		
	double sigma_y_max = 10;
	double sigma_w_max = 100;
	double length_scale_max = 100;
		
	//double[][] weight_cov;
		
	double hyper_param_update_freq = 100*w_size;
		
	Random rand = new Random();
	
	public GPWindowedBase(boolean map2fs, int input_width) {
		super(map2fs, input_width);
		
		k = new double[w_size][w_size]; 
		k_inv = new double[w_size][w_size];
		
		hyperparams = new double[2+input_width];
		hyperparams[0] = rand.nextDouble()*sigma_y_max;
		hyperparams[1] = rand.nextDouble()*sigma_w_max;
		
		for(int ctr = 0; ctr < input_width; ctr++) {
			hyperparams[2+ctr] = rand.nextDouble()*length_scale_max;
		}
		
		a = 1/(hyperparams[0]*hyperparams[0]);
		b = 1/(hyperparams[1]*hyperparams[1]);
		
		
		
		for(int ctr = 0; ctr < w_size; ctr++) {
			for(int ctr2 = 0; ctr2 < w_size; ctr2++) {
				if(ctr == ctr2) {
					k[ctr][ctr2] = (1/b + 1/a);
					k_inv[ctr][ctr2] = 1/(1/b + 1/a);
				}
				else {
					k[ctr][ctr2] = 0;
					k_inv[ctr][ctr2] = 0;
				}
			}
		}
	}
	
	protected void set_gradients(double[] gradient, double[][] y) throws Exception {
		
		double[][] derivative_cov_matrix = new double[w_size][w_size];
		
		// gradient sigma_w
		for(int ctr = 0; ctr < w_size; ctr++) {
			for(int ctr2 = 0; ctr2 < w_size; ctr2++) {
				if(ctr == ctr2) derivative_cov_matrix[ctr][ctr2] = 2.0*hyperparams[1]*hyperparams[1]; //2.0*hyperparams[1]; 
				else derivative_cov_matrix[ctr][ctr2] = (2.0)*k[ctr][ctr2]; // (2.0/hyperparams[1])*k[ctr][ctr2];
			}
		}
					
		gradient[1] = -0.5*MatrixOp.trace(MatrixOp.mult(k_inv, derivative_cov_matrix)) + 0.5*MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(y), k_inv), derivative_cov_matrix), k_inv), y)[0][0];
					
		// gradient sigma_y
		for(int ctr = 0; ctr < w_size; ctr++) {
			for(int ctr2 = 0; ctr2 < w_size; ctr2++) {
				if(ctr == ctr2) derivative_cov_matrix[ctr][ctr2] = 2*hyperparams[0]*hyperparams[0]; 
				else derivative_cov_matrix[ctr][ctr2] = 0;
			}
		}
					
		gradient[0] = 0; //-0.5*MatrixOp.trace(MatrixOp.mult(k_inv, derivative_cov_matrix)) + 0.5*MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(y), k_inv), derivative_cov_matrix), k_inv), y)[0][0];
					
		// lengthscale gradients
		for(int ctr3 = 2; ctr3 < hyperparams.length; ctr3++) {
			for(int ctr = 0; ctr < w_size; ctr++) {
				for(int ctr2 = 0; ctr2 < w_size; ctr2++) {
					if(ctr == ctr2) derivative_cov_matrix[ctr][ctr2] = 0;
					else {
						double dif = dp_window[(w_start + ctr) % w_size][ctr3-2][0] - dp_window[(w_start + ctr2) % w_size][ctr3-2][0];
						derivative_cov_matrix[ctr][ctr2] = (((dif*dif))/(Math.pow(hyperparams[ctr3], 2)))*k[ctr][ctr2]; //(((dif*dif))/(Math.pow(hyperparams[ctr3], 3)))*k[ctr][ctr2];
					}
				}
			}
			gradient[ctr3] = -0.5*MatrixOp.trace(MatrixOp.mult(k_inv, derivative_cov_matrix)) + 0.5*MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(y), k_inv), derivative_cov_matrix), k_inv), y)[0][0];
		}
		
		
	}
	
//	protected void update_hyperparams_basic_ga(double[][] responses_minus_mean_vector, double init_likhood) throws Exception {
//		while(init_likhood == Double.NaN || init_likhood == Double.NEGATIVE_INFINITY || init_likhood == Double.POSITIVE_INFINITY) {
//			hyperparams[0] = rand.nextDouble()*sigma_y_max;
//			hyperparams[1] = rand.nextDouble()*sigma_w_max;
//			
//			for(int ctr = 0; ctr < feature_count; ctr++) {
//				hyperparams[2+ctr] = rand.nextDouble()*length_scale_max;
//			}
//			
//			a = 1/(hyperparams[0]*hyperparams[0]);
//			b = 1/(hyperparams[1]*hyperparams[1]);
//			
//			recompute_k();
//			recompute_k_inv();
//			
//			init_likhood = get_likhood(responses_minus_mean_vector);
//		}
//		
//		// we have a, b and length-scales as hyperparams to tune
//		double[] gradient = new double[hyperparams.length];
//		double[] log_hyperparams = new double[hyperparams.length];
//				
//		for(int ctr = 0; ctr < hyperparams.length; ctr++)
//			log_hyperparams[ctr] = Math.log(hyperparams[ctr]);
//				
//		// GA params
//		double step_size;
//		double decay = 0.7;
//		double[] prev_gradient = new double[hyperparams.length];
//		double min_step_size = Double.MIN_NORMAL;
//		double optima_gradient_threshold = 0.0001;
//		int max_it_count = 1000;
//				
//		System.out.println("tuning...");
//		int it_count = 0;
//		
//		double cur_likhood = init_likhood;
//		
//		step_size = init_step_size;
//		
//		while(it_count++ < max_it_count) {
//			
//			set_gradients(gradient, responses_minus_mean_vector);
//			
//			optima_found = true;
//			for(int ctr = 0; ctr < hyperparams.length; ctr++) {
//				if(Math.abs(gradient[ctr]) > optima_gradient_threshold) {
//					optima_found = false;
//					break; 
//				}
//			}
//			
//			if(optima_found) {
//				System.out.println(it_count + " iterations after gradients at approx. optima:");
//				for(int ctr = 0; ctr < hyperparams.length; ctr++)
//					System.out.println(gradient[ctr]);
//				System.out.println();
//				break;
//			}
//			
//			boolean stuck = true;
//			for(int ctr = 0; ctr < hyperparams.length; ctr++) {
//				if(prev_gradient[ctr] != gradient[ctr]) {
//					stuck = false;
//					break;
//				}
//			}
//			
//			if(stuck) break;
//		
//	}
	
	protected void update_hyperparams_steepestasc(double[][] responses_minus_mean_vector, double init_likhood) throws Exception {
		
		boolean is_not_psd = false;
		
		while(is_not_psd || Double.isNaN(init_likhood) || Double.isInfinite(init_likhood)) {
			hyperparams[0] = rand.nextDouble()*sigma_y_max;
			hyperparams[1] = rand.nextDouble()*sigma_w_max;
			
			for(int ctr = 0; ctr < feature_count; ctr++) {
				hyperparams[2+ctr] = rand.nextDouble()*length_scale_max;
			}
			
			a = 1/(hyperparams[0]*hyperparams[0]);
			b = 1/(hyperparams[1]*hyperparams[1]);
			
			recompute_k();
			recompute_k_inv();
			
			try {
				recompute_k();
				recompute_k_inv();
			}
			catch (Exception e) {
				is_not_psd = true;
				continue;
			}
			
			is_not_psd = false;
			
			init_likhood = get_likhood(responses_minus_mean_vector);
		}
		
		// we have a, b and length-scales as hyperparams to tune
		double[] gradient = new double[hyperparams.length];
		double[] log_hyperparams = new double[hyperparams.length];
				
		for(int ctr = 0; ctr < hyperparams.length; ctr++)
			log_hyperparams[ctr] = Math.log(hyperparams[ctr]);
				
		// gradient ascent using RPROP
				
		// SteepestGradient params
		boolean optima_found = false;
		double[] prev_gradient = new double[hyperparams.length];
		
		double optima_gradient_threshold = 5;
		double min_gradient_delta = 0;
		int max_it_count = 10000;
				
		System.out.println("tuning...");
		int it_count = 0;
		
		double cur_likhood = init_likhood;
		
		while(it_count++ < max_it_count) {
			
			set_gradients(gradient, responses_minus_mean_vector);
			
			// optima check
			optima_found = true;
			for(int ctr = 0; ctr < gradient.length; ctr++) {
				if(Math.abs(gradient[ctr]) > optima_gradient_threshold) {
					optima_found = false;
					break; 
				}
			}
			
			if(optima_found) {
				System.out.println(it_count + " iterations after gradients at approx. optima:");
				for(int ctr = 0; ctr < hyperparams.length; ctr++)
					System.out.println(gradient[ctr]);
				System.out.println();
				break;
			}
			
			// stuck check
			boolean stuck = true;
			for(int ctr = 0; ctr < hyperparams.length; ctr++) {
				if(Math.abs(prev_gradient[ctr] - gradient[ctr]) > min_gradient_delta) {
					stuck = false;
					break;
				}
			}
			
			if(stuck) {
				cur_likhood = Double.NaN;
				it_count = 0;
				is_not_psd = false;
				
				while(is_not_psd || Double.isNaN(cur_likhood) || Double.isInfinite(cur_likhood)) {
					hyperparams[0] = rand.nextDouble()*sigma_y_max;
					hyperparams[1] = rand.nextDouble()*sigma_w_max;
					
					for(int ctr = 0; ctr < feature_count; ctr++) {
						hyperparams[2+ctr] = rand.nextDouble()*length_scale_max;
					}
					
					a = 1/(hyperparams[0]*hyperparams[0]);
					b = 1/(hyperparams[1]*hyperparams[1]);
					
					try {
						recompute_k();
						recompute_k_inv();
					}
					catch (Exception e) {
						is_not_psd = true;
						continue;
					}
					
					is_not_psd = false;
					cur_likhood = get_likhood(responses_minus_mean_vector);
				}
				
				for(int ctr = 0; ctr < hyperparams.length; ctr++) {
					prev_gradient[ctr] = 0;
					log_hyperparams[ctr] = Math.log(hyperparams[ctr]);
					System.out.print(hyperparams[ctr] + " / ");
				}
				System.out.println();
				
				continue;
				
			}
			
			// step size adaptation
			
			double min_step_size = 10E-15;
			double accelerator = 1.1;
			double step_size = min_step_size;
			int step_count = 0;
			
			while(true) {
				
				for(int ctr = 0; ctr < hyperparams.length; ctr++) {
					log_hyperparams[ctr] += step_size*gradient[ctr];
					hyperparams[ctr] = Math.pow(Math.E, log_hyperparams[ctr]);
				}
				
				a = 1/(hyperparams[0]*hyperparams[0]);
				b = 1/(hyperparams[1]*hyperparams[1]);
				
				try {
					recompute_k();
					recompute_k_inv();
				}
				catch (Exception e) {
					
					for(int ctr = 0; ctr < hyperparams.length; ctr++) {
						log_hyperparams[ctr] -= step_size*gradient[ctr];
						hyperparams[ctr] = Math.pow(Math.E, log_hyperparams[ctr]);
					}
					
					a = 1/(hyperparams[0]*hyperparams[0]);
					b = 1/(hyperparams[1]*hyperparams[1]);
						
					recompute_k();
					recompute_k_inv();
						
					cur_likhood = get_likhood(responses_minus_mean_vector);
						
					break;
				}
				
				double new_likhood = get_likhood(responses_minus_mean_vector);
				
				if(!Double.isInfinite(new_likhood) && !Double.isNaN(new_likhood) && new_likhood >= cur_likhood) {
					step_size *= accelerator;
					cur_likhood = new_likhood;
					step_count++;
				}
				else {
					for(int ctr = 0; ctr < hyperparams.length; ctr++) {
						log_hyperparams[ctr] -= step_size*gradient[ctr];
						hyperparams[ctr] = Math.pow(Math.E, log_hyperparams[ctr]);
					}
					
					a = 1/(hyperparams[0]*hyperparams[0]);
					b = 1/(hyperparams[1]*hyperparams[1]);
						
					recompute_k();
					recompute_k_inv();
						
					cur_likhood = get_likhood(responses_minus_mean_vector);
						
					break;
				}	
			}
			
			System.out.println("current_likelyhood: " + cur_likhood + ", " + step_count);
			
			for(int ctr = 0; ctr < hyperparams.length; ctr++) {
				prev_gradient[ctr] = gradient[ctr];
			}
			
		}
		
	}
	
	protected void update_hyperparams_rprop(double[][] responses_minus_mean_vector, double init_likhood) throws Exception {
		
		while(init_likhood == Double.NaN || init_likhood == Double.NEGATIVE_INFINITY || init_likhood == Double.POSITIVE_INFINITY) {
			hyperparams[0] = rand.nextDouble()*sigma_y_max;
			hyperparams[1] = rand.nextDouble()*sigma_w_max;
			
			for(int ctr = 0; ctr < feature_count; ctr++) {
				hyperparams[2+ctr] = rand.nextDouble()*length_scale_max;
			}
			
			a = 1/(hyperparams[0]*hyperparams[0]);
			b = 1/(hyperparams[1]*hyperparams[1]);
			
			recompute_k();
			recompute_k_inv();
			
			init_likhood = get_likhood(responses_minus_mean_vector);
		}
		
		// we have a, b and length-scales as hyperparams to tune
		
		double[] gradient = new double[hyperparams.length];
		double[] log_hyperparams = new double[hyperparams.length];
		
		for(int ctr = 0; ctr < hyperparams.length; ctr++)
			log_hyperparams[ctr] = Math.log(hyperparams[ctr]);
		
		// gradient ascent using RPROP
		
		// RPROP params
		double min_step_size = Double.MIN_NORMAL;
		double max_step_size = 1000;
		double init_step_size = 1;
		double step_size;
		double accelerator = 1.5;
		double decay = 0.7;
		double[] prev_gradient = new double[hyperparams.length];
		boolean optima_found = false;
		double optima_gradient_threshold = 0.01;
		int max_it_count = 1000;
		
		System.out.println("tuning...");
		
		for(int ctr = 0; ctr < hyperparams.length; ctr++)
			prev_gradient[ctr] = 0;
		
		step_size = init_step_size;
		int it_count = 0;
		
		while(!optima_found && it_count < max_it_count) {
			
			System.out.println("Iteration #" + ++it_count);
			
			set_gradients(prev_gradient, responses_minus_mean_vector);
			
			boolean illegal_gradient = false;
			for(int ctr = 0; ctr < gradient.length; ctr++) {
				if(Double.isInfinite(gradient[ctr]) || Double.isNaN(gradient[ctr]) || Math.abs(gradient[ctr]) > 10E12) {
					illegal_gradient = true;
					break;
				}
			}
			
			if(illegal_gradient) {
				for(int ctr = 0; ctr < hyperparams.length; ctr++) {
					log_hyperparams[ctr] -= step_size*prev_gradient[ctr];  //backtracking
					hyperparams[ctr] = Math.pow(Math.E, log_hyperparams[ctr]);
					prev_gradient[ctr] = 0;
				}
				step_size = Math.max(step_size*decay, min_step_size);
				continue;
			}
			
			boolean stuck = true;
			for(int ctr = 0; ctr < hyperparams.length; ctr++) {
				if(prev_gradient[ctr] != gradient[ctr]) {
					stuck = false;
					break;
				}
			}
			
			if(stuck) break;
			
			// ascending with step size adaptation
				
			if(isZeroVec(prev_gradient)) {
				for(int ctr = 0; ctr < hyperparams.length; ctr++) {
					log_hyperparams[ctr] += step_size*gradient[ctr];
					prev_gradient[ctr] = gradient[ctr];
				}
			}
			else {
				double degrees_btw_gradient_vectors = get_degrees(gradient, prev_gradient);
				if(degrees_btw_gradient_vectors < Math.PI/2) {
					step_size = Math.min(step_size*accelerator, max_step_size);
					for(int ctr = 0; ctr < hyperparams.length; ctr++) {
						log_hyperparams[ctr] += step_size*gradient[ctr];
						prev_gradient[ctr] = gradient[ctr];
					}
				}
				else if(degrees_btw_gradient_vectors >= Math.PI/2) {
					for(int ctr = 0; ctr < hyperparams.length; ctr++) {
						log_hyperparams[ctr] -= step_size*prev_gradient[ctr];  //backtracking
						hyperparams[ctr] = Math.pow(Math.E, log_hyperparams[ctr]);
						prev_gradient[ctr] = 0;
					}
					step_size = Math.max(step_size*decay, min_step_size);
					continue;
				}
				else;
			}
			
			for(int ctr = 0; ctr < hyperparams.length; ctr++) {
				hyperparams[ctr] = Math.pow(Math.E, log_hyperparams[ctr]);
				System.out.println((Math.abs(gradient[ctr]) < optima_gradient_threshold ? "*" : "") + "param " + ctr + " " + gradient[ctr] + " / " + log_hyperparams[ctr] + " / " + hyperparams[ctr] + " / " + step_size);
			}
			
			boolean illegal_params = false;
			for(int ctr = 0; ctr < hyperparams.length; ctr++) {
				if(Double.isInfinite(hyperparams[ctr]) || Double.isNaN(hyperparams[ctr])) {
					illegal_params = true;
					break;
				}
			}
		
			if(illegal_params) {
				for(int ctr = 0; ctr < hyperparams.length; ctr++) {
					log_hyperparams[ctr] -= step_size*prev_gradient[ctr];  //backtracking
					hyperparams[ctr] = Math.pow(Math.E, log_hyperparams[ctr]);
					prev_gradient[ctr] = 0;
				}
				step_size = Math.max(step_size*decay, min_step_size);
				continue;
			}
			
			a = 1/(hyperparams[0]*hyperparams[0]);
			b = 1/(hyperparams[1]*hyperparams[1]); 
			
			try {
				recompute_k();
				recompute_k_inv();
			}
			catch (Exception e) {
				for(int ctr = 0; ctr < hyperparams.length; ctr++) {
					log_hyperparams[ctr] -= step_size*prev_gradient[ctr];  //backtracking
					hyperparams[ctr] = Math.pow(Math.E, log_hyperparams[ctr]);
					prev_gradient[ctr] = 0;
				}
				step_size = Math.max(step_size*decay, min_step_size);
				continue;
			}
			
			optima_found = true;
			for(int ctr = 0; ctr < gradient.length; ctr++) {
				if(Math.abs(gradient[ctr]) > optima_gradient_threshold) {
					optima_found = false;
					break; 
				}
			}
			
			if(optima_found) {
				System.out.println(it_count + " iterations after gradients at approx. optima:");
				for(int ctr = 0; ctr < hyperparams.length; ctr++)
					System.out.println(gradient[ctr]);
			}
			
			System.out.println();
			
			for(int ctr = 0; ctr < hyperparams.length; ctr++) {
				prev_gradient[ctr] = gradient[ctr];
			}
			
		}
		
		System.out.println("post-tuning parameters");
		for(int ctr = 0; ctr < hyperparams.length; ctr++) {
			System.out.println("parameter " + ctr + " / " + log_hyperparams[ctr] + " / " + hyperparams[ctr]);
		}
			
		System.out.println("-------------------------");
		
	}

	protected double get_degrees(double[] vec1, double[] vec2) {
		
		double vec1_length = 0, vec2_length = 0;
		double vec1_dot_vec2 = 0;
		
		for(int ctr = 0; ctr < vec1.length; ctr++) {
			vec1_length += vec1[ctr]*vec1[ctr];
			vec2_length += vec2[ctr]*vec2[ctr];
			vec1_dot_vec2 += vec1[ctr]*vec2[ctr];
		}
		
		vec1_length = Math.sqrt(vec1_length);
		vec2_length = Math.sqrt(vec2_length);
		
		double cos_degrees = vec1_dot_vec2/(vec1_length*vec2_length);
		
		if(Math.abs(cos_degrees - 1) < 0.000001) cos_degrees = 1;
		
		double degrees = Math.acos(cos_degrees);
		
		System.out.println(vec1_length + " " + vec2_length + " " + vec1_dot_vec2 + " " + cos_degrees + " " + degrees);
		
		return degrees;
	}

	protected boolean isZeroVec(double[] vector) {
		for(int ctr = 0; ctr < vector.length; ctr++)
			if(vector[ctr] != 0) return false;
		
		return true;
	}
	
	protected double get_likhood(double[][] responses_minus_mean_vector) throws Exception {
		double complexity_penalty;
		try {
			complexity_penalty = -0.5*Math.log(MatrixOp.get_det_of_psd_matrix(k));
		}
		catch (Exception e) {
			complexity_penalty = Double.NEGATIVE_INFINITY;
		}
		
		double data_fit = -0.5*MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(responses_minus_mean_vector), k_inv), responses_minus_mean_vector)[0][0];
		double constant = -(n/2.0)*Math.log(Math.PI*2);
		
		return complexity_penalty + data_fit + constant;
//		return data_fit;
	}
	
	protected void recompute_k() throws Exception {
		//recomputing covariance matrix and its inverse
		//assuming sliding is already enabled
		
		if(a == 0 || b == 0) throw new Exception("Illegal precision value(s)");
		
		for(int ctr = 0; slide && ctr < w_size; ctr++) {
			for(int ctr2 = 0; ctr2 < w_size; ctr2++) {
				k[ctr][ctr2] = kernel_func(dp_window[(w_start+ctr) % w_size], dp_window[(w_start+ctr2) % w_size]);
			}
		}
	}
	
	protected void recompute_k_inv() throws Exception {
		k_inv = MatrixOp.fast_invert_psd(k);
	}

	protected double kernel_func(double[][] dp1, double dp2[][]) {
		
		double kernel_measure;
		double sum;
		
		if(dp1 == null) {
			sum = 0;
			
			for(int ctr = 0; ctr < dp2.length; ctr++) {
				double dif = dp2[ctr][0];
				sum += (dif*dif)/(hyperparams[ctr+2]*hyperparams[ctr+2]);
			}
			
			kernel_measure = Math.pow(Math.E, -0.5*sum)/b;
		}
		else {
			int dim = dp1.length;
			sum = 0;
			
			for(int ctr = 0; ctr < dim; ctr++) {
				double dif = dp1[ctr][0]- dp2[ctr][0];
				sum += (dif*dif)/(hyperparams[ctr+2]*hyperparams[ctr+2]);
			}
			
			if(sum == 0) kernel_measure = 1/b + 1/a;
			else kernel_measure = Math.pow(Math.E, -0.5*sum)/b;
		}
		
		return kernel_measure;
	}
	
//	private void print_marginal_likelihood_vals(double[][] responses_matrix, double current_mlik) throws Exception {
//		
//		double min_sigma_y = 0.1;
//		double max_sigma_y = 2.0;
//		double step_sigma_y = 0.2;
//		
//		double min_sigma_w = 0.5;
//		double max_sigma_w = 5;
//		double step_sigma_w = 0.25;
//				
//		double min_lengthscale = 45;
//		double max_lengthscale = 50;
//		double length_scale_step = 0.5;
//		
//		double step_number = (max_lengthscale - min_lengthscale) / length_scale_step;
//		double total_steps = (int) Math.pow(step_number, feature_count);
//		
//		double[] optimum_hyperparams = new double[feature_count+2];
//		
//		for(int ctr = 0; ctr < feature_count+2; ctr++)
//			optimum_hyperparams[ctr] = hyperparams[ctr];
//		
//		for(int cur_step = 0 ; cur_step < total_steps; cur_step++) {
//			for (double cur_sigma_y = min_sigma_y; cur_sigma_y <= max_sigma_y; cur_sigma_y += step_sigma_y) {
//				for (double cur_sigma_w = min_sigma_w; cur_sigma_w <= max_sigma_w; cur_sigma_w += step_sigma_w) {
//					
//					double temp = cur_step;
//					
//					for(int ctr = feature_count-1; ctr >= 0; ctr--) {
//						double cur_level = Math.pow(step_number, ctr);
//						double cur_number = Math.floor(temp/cur_level); 
//						hyperparams[ctr+2] = cur_number*length_scale_step + min_lengthscale;
//						temp -= cur_number*cur_level;
//					}
//					
//					hyperparams[0] = cur_sigma_y;
//					hyperparams[1] = cur_sigma_w;
//					
//					a = 1/(hyperparams[0]*hyperparams[0]);
//					b = 1/(hyperparams[1]*hyperparams[1]);
//					
//					recompute_k();
//					
//					try {
//						recompute_k_inv();
//					}
//					catch (Exception e) {
//						//e.printStackTrace();
//						continue;
//					}
//					
//					double complexity_penalty = -0.5*Math.log(MatrixOp.get_det_of_psd_matrix(k));
//					double data_fit = -0.5*MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(responses_matrix), k_inv), responses_matrix)[0][0];
//					double constant = -(n/2.0)*Math.log(Math.PI*2);
//					
//					double marginal_lhood = complexity_penalty + data_fit + constant;   
//					
//					if(current_mlik < marginal_lhood) {
//						current_mlik = marginal_lhood;
//						
//						System.out.print(hyperparams[0] + ", " + hyperparams[1]);
//						
//						for(int ctr = 0; ctr < feature_count; ctr++)
//								System.out.print(", " + hyperparams[ctr+2]);
//						
//						System.out.println(" -> " + marginal_lhood + " = " + complexity_penalty + " + " + data_fit + " + " + constant);
//						
//						for(int ctr = 0; ctr < feature_count+2; ctr++)
//							optimum_hyperparams[ctr] = hyperparams[ctr];
//						
//					}
//				}
//			}
//		}
//		
//		for(int ctr = 0; ctr < feature_count+2; ctr++)
//			hyperparams[ctr] = optimum_hyperparams[ctr];
//		
//		a = 1/(optimum_hyperparams[0]*optimum_hyperparams[0]);
//		b = 1/(optimum_hyperparams[1]*optimum_hyperparams[1]);
//		
//		recompute_k();
//		recompute_k_inv();
//	}
}
