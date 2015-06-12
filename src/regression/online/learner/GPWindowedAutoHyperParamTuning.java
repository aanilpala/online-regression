package regression.online.learner;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import regression.online.model.Prediction;
import regression.online.util.CholeskyDecom;
import regression.online.util.MatrixOp;
import regression.online.util.MatrixPrinter;


public class GPWindowedAutoHyperParamTuning extends WindowRegressor {

	double[][] k; // gram matrix
	double[][] k_inv; // inverse gram matrix	
	
	// hyperparams
	double a; //measurement_precision;
	double b; //weight_precision;
	double[] hyperparams; // length-scale array;
	
	double sigma_y_max = 100;
	double sigma_w_max = 100;
	double length_scale_max = 200;
	
	double[][] weight_cov;
	
	double hyper_param_update_freq = 100*w_size;
	
	Random rand = new Random();
	
	public GPWindowedAutoHyperParamTuning(int input_width, double signal_stddev, double weight_stddev) {
		super(false, input_width);
		
		name = this.getClass().getName();
		
		k = new double[w_size][w_size]; 
		k_inv = new double[w_size][w_size];
		
		hyperparams = new double[2+input_width];
		
		hyperparams[0] = signal_stddev;
		hyperparams[1] = weight_stddev;
		
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
	
	@Override
	public Prediction predict(double[][] dp) throws Exception {
		
		double[][] spare_column = new double[w_size][1];
		
		count_dps_in_window();
		
		if(slide) {
			for(int ctr = 0; ctr < w_size; ctr++) {
				spare_column[ctr][0] = kernel_func(dp_window[(w_start + ctr) % w_size], dp);
			}
		}
		else {
			int ctr = 0;
			for(; ctr < n; ctr++) {
				spare_column[w_size - 1 - ctr][0] = kernel_func(dp_window[w_end - 1 - ctr], dp);
			}
			for(; ctr < w_size - 1; ctr++) {
				spare_column[w_size - 1 - ctr][0] = 0;
			}
		}
		
		double[][] temp_column = MatrixOp.mult(MatrixOp.transpose(spare_column), k_inv);
		
		double y = 0;
		
		for(int ctr = 0; ctr < n; ctr++)
			y += temp_column[0][w_size - n + ctr]*responses[(w_start + ctr) % w_size][0];
		
		double dif = kernel_func(dp, dp) -  MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(spare_column), k_inv), spare_column)[0][0];
		double predictive_deviance = 0;
		
		if(Math.abs(dif) < 0.0000001) predictive_deviance = 0;
		else predictive_deviance = Math.sqrt(dif);
		
		return new Prediction(y, predictive_deviance);
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
		
		double[][] shrunk = new double[w_size-1][w_size-1];
		double[][] shrunk_inv = new double[w_size-1][w_size-1];
		double[][] spare_column = new double[w_size-1][1];
		double spare_var;
		
		// shrinking
		// obtaining k_minus_hat
		for(int ctr = 1; ctr < w_size; ctr++)
			shrunk[ctr-1] = Arrays.copyOfRange(k[ctr], 1, w_size);
		
		for(int ctr = 1; ctr < w_size; ctr++) 
			shrunk_inv[ctr-1] = Arrays.copyOfRange(k_inv[ctr], 1, w_size);
		
		for(int ctr = 1; ctr < w_size; ctr++)
			spare_column[ctr-1][0] = k_inv[ctr][0];
		
		spare_var = k_inv[0][0];
		
		shrunk_inv = MatrixOp.mat_add(shrunk_inv, MatrixOp.scalarmult(MatrixOp.mult(spare_column, MatrixOp.transpose(spare_column)), -1.0/spare_var));
		
		count_dps_in_window();
		
		if(slide) {
			for(int ctr = 1; ctr < w_size; ctr++) {
				spare_column[ctr-1][0] = kernel_func(dp_window[(w_start + ctr) % w_size], dp);
			}
		}
		else {
			int ctr = 0;
			for(; ctr < n; ctr++) {
				spare_column[w_size - 2 - ctr][0] = kernel_func(dp_window[w_end - 1 - ctr], dp);
			}
			for(; ctr < w_size - 1; ctr++) {
				spare_column[w_size - 2 - ctr][0] = 0; //kernel_func(null, dp);;
			}
		}
		
		for(int ctr = 1; ctr < w_size; ctr++) {
			for(int ctr2 = 1; ctr2 < w_size; ctr2++) {
				k[ctr-1][ctr2-1] = shrunk[ctr-1][ctr2-1];
			}
		}
		
		for(int ctr = 1; ctr < w_size; ctr++) {
			k[ctr-1][w_size-1] = spare_column[ctr-1][0];
			k[w_size-1][ctr-1] = spare_column[ctr-1][0];
		}
		
		spare_var = kernel_func(dp, dp);
		k[w_size-1][w_size-1] = spare_var;
		
		// computing new k_inv
		
		spare_var = 1.0/(spare_var - MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(spare_column), shrunk_inv), spare_column)[0][0]);
				
		double[][] temp_upper_left = MatrixOp.mult(shrunk_inv, MatrixOp.identitiy_add(MatrixOp.scalarmult(MatrixOp.mult(MatrixOp.mult(spare_column, MatrixOp.transpose(spare_column)), MatrixOp.transpose(shrunk_inv)), spare_var), 1));
				
		spare_column = MatrixOp.scalarmult(MatrixOp.mult(shrunk_inv, spare_column), -1*spare_var);		
		
		for(int ctr = 1; ctr < w_size; ctr++) {
			for(int ctr2 = 1; ctr2 < w_size; ctr2++) {
				k_inv[ctr-1][ctr2-1] = temp_upper_left[ctr-1][ctr2-1];
			}
		}
		
		for(int ctr = 1; ctr < w_size; ctr++) {
			k_inv[ctr-1][w_size-1] = spare_column[ctr-1][0];
			k_inv[w_size-1][ctr-1] = spare_column[ctr-1][0];
		}
		
		k_inv[w_size-1][w_size-1] = spare_var;
		
		// sliding the dp_window
		
		for(int ctr = 0; ctr < feature_count; ctr++) {
			dp_window[w_end][ctr][0] = dp[ctr][0];
		}
		
		responses[w_end][0] = y;
		
		if(slide) {
			w_start = (w_start + 1) % w_size;
			w_end = (w_end + 1) % w_size;
		}
		else {
			w_end = (w_end + 1) % w_size;
			
			if(w_start == w_end) slide = true;
		}
		
//		System.out.println("post-update");
//		MatrixPrinter.print_matrix(k);
//		System.out.println("post-update_inv");
//		MatrixPrinter.print_matrix(k_inv);		
//		
//		System.out.println("--------------------------------");
		
//		try {
//			System.in.read();
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
		
		update_count++;
		
		if(slide && ((update_count - w_size) % hyper_param_update_freq == 0)) { 
			
			count_dps_in_window();
			
			double[][] responses_matrix = new double[w_size][1];
			
			// creating y vector
			
			if(slide) {
				for(int ctr = 0; ctr < w_size; ctr++)
					responses_matrix[ctr][0] = responses[(w_start + ctr) % w_size][0];
				}
			else {  // means early  hyperparameter tuning
				int ctr = 0;
				for(; ctr < n; ctr++) {
					responses_matrix[w_size - 1 - ctr][0] = responses[w_end - 1 - ctr][0];
				}
				for(; ctr < w_size - 1; ctr++) {
					responses_matrix[w_size - 1 - ctr][0] = 0;
				}
			}
			
			double complexity_penalty = 0;
			try {
				complexity_penalty = -0.5*Math.log(MatrixOp.get_det_of_psd_matrix(k));
			}
			catch (Exception e) {
				complexity_penalty = Double.NEGATIVE_INFINITY;
			}
			
			double data_fit = -0.5*MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(responses_matrix), k_inv), responses_matrix)[0][0];
			double constant = -(n/2.0)*Math.log(Math.PI*2);
			
			double marginal_lhood = complexity_penalty + data_fit + constant;
			
//			System.out.println("pre-opt covariance matrix");
//			MatrixPrinter.print_matrix(k);
//			System.out.println("pre-opt covariance matrix");
//			MatrixPrinter.print_matrix(k_inv);		
//			
//			System.out.println("--------------------------------");
			
			System.out.println("PRE-OPTIMIZATION ML :  -> " + marginal_lhood + " = " + complexity_penalty + " + " + data_fit + " + " + constant);
			
			//print_marginal_likelihood_vals(responses_matrix, marginal_lhood);
			
			update_hyperparams();
			
			data_fit = -0.5*MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(responses_matrix), k_inv), responses_matrix)[0][0];
			constant = -(n/2.0)*Math.log(Math.PI*2);
			
			marginal_lhood = complexity_penalty + data_fit + constant;
			
			System.out.println("POST-OPTIMIZATION ML :  -> " + marginal_lhood + " = " + complexity_penalty + " + " + data_fit + " + " + constant);
		}
	}

	private int getIndexForDp(double[][] dp) {
		
		count_dps_in_window();
		
		for(int ctr = 0; ctr < n; ctr++)
			if(MatrixOp.isEqual(dp, dp_window[(w_start + ctr) % w_size])) return (w_start + ctr) % w_size;
		
		return -1;
	}

	private void set_weights_cov_matrix() throws Exception {
		
		if(!slide) throw new Exception("Slide should be enabled for this op!");
		
		count_dps_in_window();
		
		double[][] design_matrix = new double[n][feature_count];
		double[][] ones = new double[n][1];
		
		for(int ptr = w_start, ctr = 0; ctr < n; ptr = (ptr + 1) % w_size, ctr++) {
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				design_matrix[ptr][ctr2] += dp_window[ptr][ctr2][0];
			}
		}
		
		for(int ctr = 0; ctr < n; ctr++)
			ones[ctr][0] = 1;
		
		design_matrix = MatrixOp.mat_subtract(design_matrix, MatrixOp.scalarmult(MatrixOp.mult(MatrixOp.mult(ones, MatrixOp.transpose(ones)), design_matrix), 1.0/n));
		weight_cov = MatrixOp.scalarmult(MatrixOp.mult(MatrixOp.transpose(design_matrix), design_matrix), 1.0/n);
		
	}
	
	private void print_marginal_likelihood_vals(double[][] responses_matrix, double current_mlik) throws Exception {
		
		double min_sigma_y = 0.1;
		double max_sigma_y = 2.0;
		double step_sigma_y = 0.2;
		
		double min_sigma_w = 0.5;
		double max_sigma_w = 5;
		double step_sigma_w = 0.25;
				
		double min_lengthscale = 45;
		double max_lengthscale = 50;
		double length_scale_step = 0.5;
		
		double step_number = (max_lengthscale - min_lengthscale) / length_scale_step;
		double total_steps = (int) Math.pow(step_number, feature_count);
		
		double[] optimum_hyperparams = new double[feature_count+2];
		
		for(int ctr = 0; ctr < feature_count+2; ctr++)
			optimum_hyperparams[ctr] = hyperparams[ctr];
		
		for(int cur_step = 0 ; cur_step < total_steps; cur_step++) {
			for (double cur_sigma_y = min_sigma_y; cur_sigma_y <= max_sigma_y; cur_sigma_y += step_sigma_y) {
				for (double cur_sigma_w = min_sigma_w; cur_sigma_w <= max_sigma_w; cur_sigma_w += step_sigma_w) {
					
					double temp = cur_step;
					
					for(int ctr = feature_count-1; ctr >= 0; ctr--) {
						double cur_level = Math.pow(step_number, ctr);
						double cur_number = Math.floor(temp/cur_level); 
						hyperparams[ctr+2] = cur_number*length_scale_step + min_lengthscale;
						temp -= cur_number*cur_level;
					}
					
					hyperparams[0] = cur_sigma_y;
					hyperparams[1] = cur_sigma_w;
					
					a = 1/(hyperparams[0]*hyperparams[0]);
					b = 1/(hyperparams[1]*hyperparams[1]);
					
					recompute_k();
					
					try {
						recompute_k_inv();
					}
					catch (Exception e) {
						//e.printStackTrace();
						continue;
					}
					
					double complexity_penalty = -0.5*Math.log(MatrixOp.get_det_of_psd_matrix(k));
					double data_fit = -0.5*MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(responses_matrix), k_inv), responses_matrix)[0][0];
					double constant = -(n/2.0)*Math.log(Math.PI*2);
					
					double marginal_lhood = complexity_penalty + data_fit + constant;   
					
					if(current_mlik < marginal_lhood) {
						current_mlik = marginal_lhood;
						
						System.out.print(hyperparams[0] + ", " + hyperparams[1]);
						
						for(int ctr = 0; ctr < feature_count; ctr++)
								System.out.print(", " + hyperparams[ctr+2]);
						
						System.out.println(" -> " + marginal_lhood + " = " + complexity_penalty + " + " + data_fit + " + " + constant);
						
						for(int ctr = 0; ctr < feature_count+2; ctr++)
							optimum_hyperparams[ctr] = hyperparams[ctr];
						
					}
				}
			}
		}
		
		for(int ctr = 0; ctr < feature_count+2; ctr++)
			hyperparams[ctr] = optimum_hyperparams[ctr];
		
		a = 1/(optimum_hyperparams[0]*optimum_hyperparams[0]);
		b = 1/(optimum_hyperparams[1]*optimum_hyperparams[1]);
		
		recompute_k();
		recompute_k_inv();
	}

	private void update_hyperparams() throws Exception {
		
		// we have a, b and length-scales as hyperparams to tune
		
		count_dps_in_window();
		
		double[][] derivative_cov_matrix = new double[w_size][w_size];
		double[][] y = new double[w_size][1];
		double[] gradient = new double[hyperparams.length];
		double[] log_hyperparams = new double[hyperparams.length];
		
		for(int ctr = 0; ctr < hyperparams.length; ctr++)
			log_hyperparams[ctr] = Math.log(hyperparams[ctr]);
		
		System.out.println("pre-tuning parameters");
		for(int ctr = 0; ctr < hyperparams.length; ctr++) {
			System.out.println("parameter " + ctr + " " + log_hyperparams[ctr] + hyperparams[ctr]);
		}
		
		// creating y vector
		
		if(slide) {
			for(int ctr = 0; ctr < w_size; ctr++)
				y[ctr][0] = responses[(w_start + ctr) % w_size][0];
		}
		else {  // means early  hyperparameter tuning
			int ctr = 0;
			for(; ctr < n; ctr++) {
				y[w_size - 1 - ctr][0] = responses[w_end - 1 - ctr][0];
			}
			for(; ctr < w_size - 1; ctr++) {
				y[w_size - 1 - ctr][0] = 0;
			}
		}
		
		// gradient ascent using RPROP
		
		
		// RPROP params
		double min_step_size = 0.000001;
		double max_step_size = 1000;
		double[] step_size = new double[hyperparams.length];
		double accelerator = 1.5;
		double decay = 0.1;
		double[] prev_gradient = new double[hyperparams.length];
		boolean optima_found = false;
		double optima_gradient_threshold = 0.01;
		int max_it_count = 100000;
		
		System.out.println("tuning...");
		int it_count = 0;
		
		for(int ctr = 0; ctr < hyperparams.length; ctr++) {
			step_size[ctr] = 0.01;
			prev_gradient[ctr] = 0;
		}
		
		//while(learning_rate > 0 && delta > min_delta) {
		while(!optima_found && it_count < max_it_count) {
			
			System.out.println("Iteration #" + ++it_count);
			
			double[][] alpha = MatrixOp.mult(k_inv, y);
			
			// tuning sigma_w
			for(int ctr = 0; ctr < w_size; ctr++) {
				for(int ctr2 = 0; ctr2 < w_size; ctr2++) {
					if(ctr == ctr2) derivative_cov_matrix[ctr][ctr2] = 2.0*hyperparams[1]*hyperparams[1]; //2.0*hyperparams[1]; 
					else derivative_cov_matrix[ctr][ctr2] = (2.0)*k[ctr][ctr2]; // (2.0/hyperparams[1])*k[ctr][ctr2];
				}
			}
			gradient[1] = 0.5*MatrixOp.trace(MatrixOp.mult(MatrixOp.mat_subtract(MatrixOp.mult(alpha, MatrixOp.transpose(alpha)), k_inv), derivative_cov_matrix)); // - 10/sigma_w;
			
			// tuning sigma_y
			for(int ctr = 0; ctr < w_size; ctr++) {
				for(int ctr2 = 0; ctr2 < w_size; ctr2++) {
					if(ctr == ctr2) derivative_cov_matrix[ctr][ctr2] = 2*hyperparams[0]*hyperparams[0]; 
					else derivative_cov_matrix[ctr][ctr2] = 0;
				}
			}
			gradient[0] = 0;// 0.5*MatrixOp.trace(MatrixOp.mult(MatrixOp.mat_subtract(MatrixOp.mult(alpha, MatrixOp.transpose(alpha)), k_inv), derivative_cov_matrix));// - 1/(hyper_params[0]);
			
			for(int ctr3 = 2; ctr3 < hyperparams.length; ctr3++) {
				for(int ctr = 0; ctr < w_size; ctr++) {
					for(int ctr2 = 0; ctr2 < w_size; ctr2++) {
						if(ctr == ctr2) {
							derivative_cov_matrix[ctr][ctr2] = 0;
							//derivative_cov_matrix[ctr][ctr2] = ((2.0/b)/(Math.pow(hyperparams[ctr3], 3)));
						}
						else {
							double dif = dp_window[(w_start + ctr) % w_size][ctr3-2][0] - dp_window[(w_start + ctr2) % w_size][ctr3-2][0];
							derivative_cov_matrix[ctr][ctr2] = (((dif*dif))/(Math.pow(hyperparams[ctr3], 2)))*k[ctr][ctr2]; //(((dif*dif))/(Math.pow(hyperparams[ctr3], 3)))*k[ctr][ctr2];
						}
					}
				}
				gradient[ctr3] = 0.5*MatrixOp.trace(MatrixOp.mult(MatrixOp.mat_subtract(MatrixOp.mult(alpha, MatrixOp.transpose(alpha)), k_inv), derivative_cov_matrix));
			}
			
			
			// ascending with step size adaptation
			for(int ctr = 0; ctr < hyperparams.length; ctr++) {
				int cur_sign = (int) Math.signum(gradient[ctr]);
				int prev_sign = (int) Math.signum(prev_gradient[ctr]);
				
				if(cur_sign*prev_sign == 1) {
					step_size[ctr] = Math.min(step_size[ctr]*accelerator, max_step_size);
					log_hyperparams[ctr] += step_size[ctr]*gradient[ctr];
					prev_gradient[ctr] = gradient[ctr];
				}
				else if(cur_sign*prev_sign == -1) {
					log_hyperparams[ctr] -= step_size[ctr]*prev_gradient[ctr];  //backtracking
					step_size[ctr] = Math.max(step_size[ctr]*decay, min_step_size);
					prev_gradient[ctr] = 0;
				}
				else if(prev_sign == 0) {
					log_hyperparams[ctr] += step_size[ctr]*gradient[ctr];
					prev_gradient[ctr] = gradient[ctr];
				}
				else { //perfect optima found for the parameter
					prev_gradient[ctr] = gradient[ctr];
				}
				
//				if(log_hyperparams[ctr] < -10) {
//					if(ctr == 0) { 
//						hyperparams[ctr] = 0.1;
//						log_hyperparams[ctr] = Math.log(hyperparams[ctr]);
//						step_size[ctr] = 0;
//						prev_gradient[ctr] = 0;
//					}
//				}
				
				hyperparams[ctr] = Math.pow(Math.E, log_hyperparams[ctr]);
				System.out.println((Math.abs(gradient[ctr]) < optima_gradient_threshold ? "*" : "") + "param " + ctr + " " + gradient[ctr] + " / " + log_hyperparams[ctr] + " / " + hyperparams[ctr] + " / " + step_size[ctr]);
				
//				try {
//					System.in.read();
//				} catch (IOException e) {
//					e.printStackTrace();
//				}
				
//				if(hyperparams[ctr] <= 0 || hyperparams[ctr] > 100000) {
//					//System.out.println("ILLEGAL!");
//					//System.out.println(gradient[ctr] + " / " + hyperparams[ctr] + " / " + step_size[ctr]);
//					
//					if(ctr == 0) hyperparams[ctr] = sigma_y_max*rand.nextDouble();
//					else if(ctr == 1) hyperparams[ctr] = sigma_w_max*rand.nextDouble();
//					else hyperparams[ctr] = length_scale_max*rand.nextDouble();
//					
//					step_size[ctr] = 0.01;
//					prev_gradient[ctr] = 0;
//				}				
			}
			
			a = 1/(hyperparams[0]*hyperparams[0]);
			b = 1/(hyperparams[1]*hyperparams[1]);
			
			//System.out.println(gradient[1] + " / " + step_size[1]); 
			
			recompute_k();
			recompute_k_inv();
			
			optima_found = true;
			for(int ctr = 0; ctr < hyperparams.length; ctr++) {
				if(Math.abs(gradient[ctr]) > optima_gradient_threshold) {
					optima_found = false;
					//break; 
				}
				else {
					step_size[ctr] = min_step_size;
				}
			}
			
			if(optima_found) {
				System.out.println(it_count + " iterations after gradients at approx. optima:");
				for(int ctr = 0; ctr < hyperparams.length; ctr++)
					System.out.println(gradient[ctr]);
			}
			
			System.out.println();
			
		}
		
		System.out.println("post-tuning parameters");
		for(int ctr = 0; ctr < hyperparams.length; ctr++) {
			System.out.println("parameter " + ctr + " / " + log_hyperparams[ctr] + " / " + hyperparams[ctr]);
		}
			
		System.out.println("-------------------------");
		
//		try {
//			System.in.read();
//		} catch (IOException e) {
//			e.printStackTrace();
//		}

	}

	private void recompute_k() {
		//recomputing covariance matrix and its inverse
		//assuming sliding is already enabled
		for(int ctr = 0; slide && ctr < w_size; ctr++) {
			for(int ctr2 = 0; ctr2 < w_size; ctr2++) {
				k[ctr][ctr2] = kernel_func(dp_window[(w_start+ctr) % w_size], dp_window[(w_start+ctr2) % w_size]);
			}
		}
	}
	
	private void recompute_k_inv() throws Exception {
		k_inv = MatrixOp.fast_invert_psd(k);
	}

	private double kernel_func(double[][] dp1, double dp2[][]) {
		
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
}
