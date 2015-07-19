package regression.test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import aima.core.learning.framework.DataSet;
import regression.online.learner.BayesianMAPForgetting;
import regression.online.learner.BayesianMAPForgetting;
import regression.online.learner.BayesianMAPWindowed;
import regression.online.learner.BayesianMLEForgetting;
import regression.online.learner.BayesianMLEForgetting;
import regression.online.learner.BayesianMLEWindowed;
import regression.online.learner.BayesianMLEWindowed;
import regression.online.learner.BayesianMAPWindowed;
import regression.online.learner.BayesianMLEWindowed;
import regression.online.learner.GPWindowedGaussianKernelAvgMean;
import regression.online.learner.GPWindowedGaussianKernelOLSMean;
import regression.online.learner.GPWindowedGaussianKernelZeroMean;
import regression.online.learner.KernelRegression;
import regression.online.learner.OnlineRegressor;
import regression.online.learner.obsolete.BayesianPredictive;
import regression.online.learner.obsolete.BayesianPredictiveMapped;
import regression.online.learner.obsolete.BayesianPredictiveWindowed;
import regression.online.learner.obsolete.BayesianPredictiveWindowedMapped;
import regression.online.model.Prediction;

public class OnlineRegressorTest {

	public OnlineRegressor reg;
	
	List<double[][]> data_points;
	List<Double> responses;
	public String ds_name;
	
	public double rmse, rmse_st, smse, smse_st;
	public double icr;
	public double aiw, saiw;
	public double apt, hpt, tpt, ptr;
	public double aut, hut, tut, utr;
	public double att, htt, ttt, ttr;
	public int tc; // tuning count
	
	private boolean st;
	
	public OnlineRegressorTest(OnlineRegressor reg, TestCase testcase) {
		
		this.data_points = testcase.data_points;
		this.responses = testcase.responses;
		
		this.st = testcase.st;
		
		if(data_points.size() != responses.size()) {
			System.out.println("Size Inconsistency");
			return;
		}
		
		this.ds_name = testcase.data_set_name;
		this.reg = reg;
		
		// initializing the stats
		this.smse = 0;
		this.rmse = 0;
		
		this.hpt = 0;
		this.tpt = 0;
		this.hut = 0;
		this.tut = 0;
		this.htt = 0;
		this.ttt = 0;
		
		this.tc = 0;
		
	}
	
	public void test(boolean dump_logs) {
		int interval_miss_count= 0;
		long accumulated_squared_error_wcd = 0;
		long accumulated_squared_error_wocd = 0;
		long accumulated_interval_width = 0;
		int burn_in_number = reg.get_burn_in_number();
		int m_n = 0;
		int interval_width_counter = 0;
		long aggregate_update_time = 0;
		long aggregate_tuning_time = 0;
		long aggregate_prediction_time = 0;
		int optimizer_run_count = 0;
		
		double m_new, m_old = 0, s_new = 0, s_old = 0; // needed for target variance computation
		long aggreagete_target_sum = 0; 
		
		FileWriter logs_fw = null;
		FileWriter prediction_plot_fw = null;
		
		if(dump_logs) {
			try {
				logs_fw = new FileWriter(Path.logs_path_online + this.reg.get_name() + "_on_" + this.ds_name + "_logs.txt");
				prediction_plot_fw = new FileWriter(Path.plots_path_online + this.reg.get_name() + "_on_" + this.ds_name + "_prediction_plot_data.txt");
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		for(int ctr = 0; ctr < data_points.size(); ctr++) {
			double[][] dp = data_points.get(ctr);
			Double response = responses.get(ctr);
			
			Prediction pred = null;
			Double err = Double.NaN;
			
			try {
				long startTime = System.nanoTime();    
				pred = reg.predict(dp);
				long elapsedTime = System.nanoTime() - startTime;
				
				if(elapsedTime/1000000.0 > this.hpt) this.hpt = elapsedTime/1000000.0;
				aggregate_prediction_time += elapsedTime;
				
				startTime = System.nanoTime();
				reg.update(dp, response, pred);
				elapsedTime = System.nanoTime() - startTime;
				
				if(reg.was_just_tuned()) {
					if(elapsedTime/1000000.0 > this.htt) this.htt = elapsedTime/1000000.0;
					aggregate_tuning_time += elapsedTime;
					optimizer_run_count++;
				}
				else {
					if(elapsedTime/1000000.0 > this.hut) this.hut = elapsedTime/1000000.0;
					aggregate_update_time += elapsedTime;
				}
				
			} 
			catch (Exception e1) {
				e1.printStackTrace();
			}
			
			if(!Double.isNaN(pred.point_prediction))
				err = Math.abs(pred.point_prediction - response);
			else
				err = pred.point_prediction;
			
			if(!Double.isNaN(pred.point_prediction)) {
				m_n++;
				if(m_n == 1) {
					m_old = m_new = response;
					s_old = 0;
				}
				else {
					m_new = m_old + (response - m_old)/((float) m_n);
					s_new = s_old + (response - m_old)*(response - m_new);
						
					//for the next iteration
					m_old = m_new;
					s_old = s_new;
				}
					
				accumulated_squared_error_wcd += Math.pow((pred.point_prediction - response)*100, 2);
				
				if(!st && (ctr % 1000 > burn_in_number))
					accumulated_squared_error_wocd += Math.pow((pred.point_prediction - response)*100, 2);
				else if(st && (ctr > burn_in_number))
					accumulated_squared_error_wocd += Math.pow((pred.point_prediction - response)*100, 2);
			}
				
				
			if(!Double.isNaN(pred.lower_bound) && !(Double.isNaN(pred.lower_bound))) {
				interval_width_counter++;
				if(pred.upper_bound < response || response < pred.lower_bound) 
					interval_miss_count++;
					
				accumulated_interval_width += (long) ((pred.upper_bound - pred.lower_bound)*100);
				
				aggreagete_target_sum += (long) (response*100000);
			}
			
			if(dump_logs) {
				try {
					String plot_data_row = "";
					for(int ctr2 = 0; ctr2 < dp.length; ctr2++)
						plot_data_row += dp[ctr2][0] + "\t";
					
					plot_data_row += pred.lower_bound + "\t" + pred.point_prediction + "\t" + pred.upper_bound + "\t";
					plot_data_row += response + "\t" + err;
					
					logs_fw.write( "target: " + response + " predicted: " + pred.toString());
					logs_fw.write("\n");
					
					prediction_plot_fw.write(plot_data_row);
					prediction_plot_fw.write("\n");
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	
		
		// computing stats
		double target_variance = s_new/(float) (m_n-1);
		double mse = (accumulated_squared_error_wcd / (m_n)) / 10000.0;
		double mse_st = (accumulated_squared_error_wocd / (m_n)) / 10000.0;
		double target_mean = (aggreagete_target_sum / interval_width_counter) / 100000.0;
		
//		System.out.print((accumulated_squared_error_wcd / (m_n)) / 10000.0);
//		System.out.print("-");
//		System.out.println(((accumulated_squared_error_wocd / (m_n-2*burn_in_number)) / 10000.0)/target_variance);
		
		smse = mse/target_variance;
		smse_st = mse_st/target_variance;
		rmse = Math.sqrt(mse);
		rmse_st = Math.sqrt(mse_st);
		icr = (1 - (interval_miss_count / (double) (interval_width_counter)))*100;
		aiw = (accumulated_interval_width / (long) interval_width_counter) / 100.0;
		saiw = aiw / target_mean;
		
		// computing time stats (in ms)
		this.tpt = aggregate_prediction_time / 1000000.0;
		this.apt = this.tpt / data_points.size();
		this.tut = aggregate_update_time / 1000000.0;
		this.aut = this.tut / (data_points.size() - optimizer_run_count);
		this.ttt = optimizer_run_count != 0 ? (aggregate_tuning_time / 1000000.0) : 0;
		this.att = optimizer_run_count != 0 ? (this.ttt / optimizer_run_count) : 0;
		
		this.ptr = tpt/(tpt + tut + ttt)*100;
		this.utr = tut/(tpt + tut + ttt)*100;
		this.ttr = ttt/(tpt + tut + ttt)*100;
		
		this.tc = optimizer_run_count;
		
		if(this.htt == 0) this.htt = 0;
		
		
		if(dump_logs) {
			try {
				logs_fw.write("SUMMARY\n");
				logs_fw.write("SMSE = " + smse + ", SMSE_ST = " + smse_st + "\n");
				logs_fw.write("RMSE = " + rmse + ", RMSE_ST = " + rmse_st + "\n");
				logs_fw.write("SAIW = " + saiw + ", AIW = " + aiw + " ICR = " + icr + "%\n");
				logs_fw.write("APT = " + apt + " msec" + ", AUT = " + aut + " msec" + ", ATT = " + att + " msec\n");
				logs_fw.write("HPT = " + hpt + " msec" + ", HUT = " + hut + " msec" + ", HTT = " + htt + " msec\n");
				logs_fw.write("TPT = " + tpt + " msec" + ", TUT = " + tut + " msec" + ", TTT = " + ttt + " msec\n");
				logs_fw.write("PTR = " + ptr + "%" + ", UTR = " + utr + "%" + ", TTR = " + ttr + "%\n");
				
				logs_fw.flush();
				logs_fw.close();
				prediction_plot_fw.flush();
				prediction_plot_fw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	private static void clean_dirs() {
		
		File logs_dir = new File(Path.logs_path_online);
		File plots_dir = new File(Path.plots_path_online);
		
		File[] log_files = logs_dir.listFiles();
		File[] plot_files = plots_dir.listFiles();
		
	    if(log_files != null) {
	        for(File f: log_files)
	            if(!f.isDirectory()) f.delete();
	    }
	    
	    if(plot_files != null) {
	        for(File f: plot_files)
	            if(!f.isDirectory()) f.delete();
	    }
		
	}

	public double getSMSE(){
		return smse;
	}
	
	public double getContainmentRate(){
		return icr;
	}
	
	public String getRegName() {
		return reg.get_name();
	}
	
	public static void main(String[] args) {
		
		clean_dirs();
		
		List<TestCase> test_cases = new ArrayList<TestCase>();
		
		File data_path = new File(Path.data_path); 
		for(File file : data_path.listFiles()) {
			if(file.isFile() && !file.isHidden()) {
				try{
					String name = file.getName();
					
					boolean is_st = name.split("\\_")[3] == "NCD" ? true : false;
					
					TestCase testcase = new TestCase(file, is_st);
					test_cases.add(testcase);
				}
				catch (Exception e) {
					e.printStackTrace();
				}
			}
		}
		
		FileWriter comparison_table_fw = null;
		
		try {
			comparison_table_fw = new FileWriter("./data/" + "comparison_table_online.csv");
			comparison_table_fw.write("NAME\t" + "TEST\t" + "SMSE\t" + "SMSE_ST\t" + "RMSE\t" + "RMSE_ST\t" + "AIW\t" + "SMIW\t" + "ICR\t" + "APT\t" + "AUT\t" + "ATT\t" + "HPT\t" + "HUT\t" + "HTT\t" + "TPT\t" + "TUT\t" + "TTT\t" + "PTR\t" + "UTR\t" + "TTR\t" + "TN\n");
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		
		HashMap<String, List<OnlineRegressorTest>> regtests = new HashMap<String, List<OnlineRegressorTest>>();
		for(TestCase testcase : test_cases) {
			List<OnlineRegressor> regs = new ArrayList<OnlineRegressor>();
			
			int input_width = testcase.input_width;
			
			int window_sizes[] = new int[]{32, 64, 128};
			double forgetting_factors[] = new double[]{0.0, 0.05, 0.1};
			
			for(double forgetting_factor : forgetting_factors) {
//				regs.add(new BayesianMLEForgetting(input_width, forgetting_factor, false));	// Ad-Hoc prediction intervals
//				regs.add(new BayesianMLEForgetting(input_width, forgetting_factor, true));		// Ad-Hoc prediction intervals
				
//				regs.add(new BayesianMAPForgetting(input_width, forgetting_factor, false, 0.1, 2));	// Ad-Hoc prediction intervals
//				regs.add(new BayesianMAPForgetting(input_width, forgetting_factor, true, 0.1, 2));		// Ad-Hoc prediction intervals
			}
			
			for(int w_size : window_sizes) {
//				regs.add(new BayesianMLEWindowed(input_width, w_size, false, -1));	// OLS asymptotic prediction intervals
//				regs.add(new BayesianMLEWindowed(input_width, w_size, true, -1));	// OLS asymptotic prediction intervals

//				regs.add(new BayesianMAPWindowed(input_width, w_size, false, 0.1, 2, -1));	// OLS asymptotic prediction intervals
				regs.add(new BayesianMAPWindowed(input_width, w_size, true, 0.1, 2, -1));	// OLS asymptotic prediction intervals

				regs.add(new KernelRegression(input_width, w_size, 1));	// Confidence Intervals instead of Prediction Intervals
				
//				regs.add(new GPWindowedGaussianKernelAvgMean(input_width, w_size, 0.1, 2, false, 1, false));				
//				regs.add(new GPWindowedGaussianKernelOLSMean(input_width, w_size, 0.1, 2, false, 1));		// OLS models the data as much as it can and the residuals are modeled by the GP Regression
//				regs.add(new GPWindowedGaussianKernelZeroMean(input_width, w_size, 0.1, 2, false, 1, false));
			}
			
			for(OnlineRegressor each : regs) {
				String cur_id = each.name;
				List<OnlineRegressorTest> cur_tests = regtests.get(cur_id);
				if(cur_tests == null) {
					cur_tests = new ArrayList<OnlineRegressorTest>();
					regtests.put(cur_id, cur_tests);
				}
				
				cur_tests.add(new OnlineRegressorTest(each, testcase));
			}	
			
			// FOR TESTING
//			break;
		}

		for(String name : regtests.keySet()) {
			
			List<OnlineRegressorTest> cur_tests = regtests.get(name);
			
			for(OnlineRegressorTest each : cur_tests) {
				each.test(true);
				try {
					comparison_table_fw.write(each.getRegName() + "\t" 
							+ each.ds_name + "\t" 
							+ each.smse + "\t"
							+ each.smse_st + "\t"
							+ each.rmse + "\t"
							+ each.rmse_st + "\t"
							+ each.aiw + "\t"
							+ each.saiw + "\t"
							+ each.icr + "\t"
							+ each.apt + "\t"
							+ each.aut + "\t"
							+ each.att + "\t"
							+ each.hpt + "\t"
							+ each.hut + "\t"
							+ each.htt + "\t"
							+ each.tpt + "\t"
							+ each.tut + "\t"
							+ each.ttt + "\t"
							+ each.ptr + "\t"
							+ each.utr + "\t"
							+ each.ttr + "\t"
							+ each.tc + "\n");
							
				} catch (IOException e) {
					e.printStackTrace();
				}
				
				// Console output
				System.out.println(each.reg.get_name() + " " + "on dataset " + each.ds_name + " performance summary");
				System.out.println("SMSE = " + each.smse + ", SMSE_ST = " + each.smse_st);
				System.out.println("RMSE = " + each.rmse + ", RMSE_ST = " + each.rmse_st);
				System.out.println("SAIW = " + each.saiw + ", AIW = " + each.aiw + ", ICR = " + each.icr + "%");
				System.out.println("APT = " + each.apt + " msec" + ", AUT = " + each.aut + " msec" + ", ATT = " + each.att + " msec");
				System.out.println("HPT = " + each.hpt + " msec" + ", HUT = " + each.hut + " msec" + ", HTT = " + each.htt + " msec");
				System.out.println("TPT = " + each.tpt + " msec" + ", TUT = " + each.tut + " msec" + ", TTT = " + each.ttt + " msec");
				System.out.println("PTR = " + each.ptr + "%" + ", UTR = " + each.utr + "%" + ", TTR = " + each.ttr + "%");
				System.out.println("TC = " + each.tc);
				System.out.println();
				
			}
			
		}
		
		try {
			comparison_table_fw.flush();
			comparison_table_fw.close();
			
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
}
