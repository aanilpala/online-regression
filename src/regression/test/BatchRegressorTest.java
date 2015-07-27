package regression.test;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import regression.offline.learner.BatchRegressor;
import regression.offline.learner.BayesianMLEBatch;
import regression.offline.learner.GPRegressionBatch;
import regression.offline.learner.KernelRegressionBatch;

public class BatchRegressorTest {

	public BatchRegressor reg;
	
	public double rmse, rmse_st, smse, smse_st;
//	public double icr;
//	public double aiw, saiw;
	public double apt, hpt, tpt, ptr;
//	public double aut, hut, tut, utr;
//	public double att, htt, ttt, ttr;
//	public int tc; // tuning count

	private TestCase testcase;

	private String ds_name;
	
//	private boolean st;
	
	public BatchRegressorTest(BatchRegressor reg, TestCase testcase) {
		
		this.testcase = testcase;
		
		if(testcase.data_points.size() != testcase.responses.size()) {
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
//		this.hut = 0;
//		this.tut = 0;
//		this.htt = 0;
//		this.ttt = 0;
		
//		this.tc = 0;
		
	}
	
	private String getRegName() {
		return reg.name;
	}
	
	public void test(boolean dump_logs) {
		long accumulated_squared_error_wcd = 0;
		long accumulated_squared_error_wocd = 0;
		int training_size = reg.training_size();
		int m_n = 0;
		long aggregate_prediction_time = 0;
		
		double m_new, m_old = 0, s_new = 0, s_old = 0; // needed for target variance computation
		
		FileWriter logs_fw = null;
		FileWriter prediction_plot_fw = null;
		
		if(dump_logs) {
			try {
				logs_fw = new FileWriter(Path.logs_path_batch + this.reg.get_name() + "_on_" + this.ds_name + "_logs.txt");
				prediction_plot_fw = new FileWriter(Path.plots_path_batch + this.reg.get_name() + "_on_" + this.ds_name + "_prediction_plot_data.txt");
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		double[][][] training_set = new double[reg.training_set_size][testcase.input_width][1];
		double[][] targets = new double[training_size][1];
		
		for(int ctr = 0; ctr < testcase.data_points.size(); ctr++) {
			double[][] dp = testcase.data_points.get(ctr);
			double response = testcase.responses.get(ctr);
			
			double pred = 0;
			double err = 0;
			
			if(ctr < training_size) {
				training_set[ctr] = dp;
				targets[ctr][0] = response;
			}
			else if(ctr == training_size) {
				try {
					reg.train(training_set, targets);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			
			if(ctr >= training_size) {
				
				long elapsedTime = 0, startTime;
				
				try {
					startTime = System.nanoTime();    
					pred = reg.predict(dp);
					elapsedTime = System.nanoTime() - startTime;
				}
				catch (Exception e) {
					e.printStackTrace();
				}
				
				if(elapsedTime/1000000.0 > this.hpt) this.hpt = elapsedTime/1000000.0;
				aggregate_prediction_time += elapsedTime;
				
				if(!Double.isNaN(pred))
					err = Math.abs(pred- response);
				else
					err = pred;
				
				if(!Double.isNaN(pred)) {
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
						
					accumulated_squared_error_wcd += Math.pow((pred - response)*100, 2);
				}
			}
			
			if(dump_logs) {
				try {
					String plot_data_row = "";
					for(int ctr2 = 0; ctr2 < dp.length; ctr2++)
						plot_data_row += dp[ctr2][0] + "\t";
					
					plot_data_row += pred + "\t" + pred + "\t" + pred+ "\t";
					plot_data_row += response + "\t" + err;
					
					logs_fw.write( "target: " + response + " predicted: " + pred);
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
		
		smse = mse/target_variance;
		smse_st = mse_st/target_variance;
		rmse = Math.sqrt(mse);
		rmse_st = Math.sqrt(mse_st);
		
		// computing time stats (in ms)
		this.tpt = aggregate_prediction_time / 1000000.0;
		this.apt = this.tpt / testcase.data_points.size();
		
		if(dump_logs) {
			try {
				logs_fw.write("SUMMARY\n");
				logs_fw.write("SMSE = " + smse + ", SMSE_ST = " + smse_st + "\n");
				logs_fw.write("RMSE = " + rmse + ", RMSE_ST = " + rmse_st + "\n");
				logs_fw.write("SAIW = " + -1.0 + ", AIW = " + -1.0 + " ICR = " + -1.0 + "%\n");
				logs_fw.write("APT = " + apt + " msec" + ", AUT = " + -1.0 + " msec" + ", ATT = " + -1.0 + " msec\n");
				logs_fw.write("HPT = " + hpt + " msec" + ", HUT = " + -1.0 + " msec" + ", HTT = " + -1.0 + " msec\n");
				logs_fw.write("TPT = " + tpt + " msec" + ", TUT = " + -1.0 + " msec" + ", TTT = " + -1.0 + " msec\n");
				logs_fw.write("PTR = " + ptr + "%" + ", UTR = " + -1.0 + "%" + ", TTR = " + -1.0 + "%\n");
				
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
		
		File logs_dir = new File(Path.logs_path_batch);
		File plots_dir = new File(Path.plots_path_batch);
		
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
			comparison_table_fw = new FileWriter("./data/" + "comparison_table_batch.csv");
			comparison_table_fw.write("NAME\t" + "TEST\t" + "SMSE\t" + "SMSE_ST\t" + "RMSE\t" + "RMSE_ST\t" + "AIW\t" + "SMIW\t" + "ICR\t" + "APT\t" + "AUT\t" + "ATT\t" + "HPT\t" + "HUT\t" + "HTT\t" + "TPT\t" + "TUT\t" + "TTT\t" + "PTR\t" + "UTR\t" + "TTR\t" + "TN\n");
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		
		HashMap<String, List<BatchRegressorTest>> regtests = new HashMap<String, List<BatchRegressorTest>>();
		for(TestCase testcase : test_cases) {
			List<BatchRegressor> regs = new ArrayList<BatchRegressor>();
			
			int input_width = testcase.input_width;
			
			int training_set_sizes[] = new int[]{32, 64, 128};
			
			for(int training_set_size : training_set_sizes) {
				regs.add(new BayesianMLEBatch(input_width, training_set_size, false));
				regs.add(new BayesianMLEBatch(input_width, training_set_size, true));
				regs.add(new KernelRegressionBatch(input_width, training_set_size));
				regs.add(new GPRegressionBatch(input_width, training_set_size, false));
			}
			
			for(BatchRegressor each : regs) {
				String cur_id = each.name;
				List<BatchRegressorTest> cur_tests = regtests.get(cur_id);
				if(cur_tests == null) {
					cur_tests = new ArrayList<BatchRegressorTest>();
					regtests.put(cur_id, cur_tests);
				}
				
				cur_tests.add(new BatchRegressorTest(each, testcase));
			}	
			
			// FOR TESTING
//			break;
		}

		for(String name : regtests.keySet()) {
			
			List<BatchRegressorTest> cur_tests = regtests.get(name);
			
			for(BatchRegressorTest each : cur_tests) {
				each.test(true);
				try {
					comparison_table_fw.write(each.getRegName() + "\t" 
							+ each.ds_name + "\t" 
							+ each.smse + "\t"
							+ each.smse_st + "\t"
							+ each.rmse + "\t"
							+ each.rmse_st + "\t"
							+ -1.0 + "\t"
							+ -1.0 + "\t"
							+ -1.0 + "\t"
							+ each.apt + "\t"
							+ -1.0 + "\t"
							+ -1.0 + "\t"
							+ each.hpt + "\t"
							+ -1.0 + "\t"
							+ -1.0 + "\t"
							+ each.tpt + "\t"
							+ -1.0 + "\t"
							+ -1.0 + "\t"
							+ each.ptr + "\t"
							+ -1.0 + "\t"
							+ -1.0 + "\t"
							+ -1 + "\n");
							
				} catch (IOException e) {
					e.printStackTrace();
				}
				
				// Console output
				System.out.println(each.reg.get_name() + " " + "on dataset " + each.ds_name + " performance summary");
				System.out.println("SMSE = " + each.smse + ", SMSE_ST = " + each.smse_st);
				System.out.println("RMSE = " + each.rmse + ", RMSE_ST = " + each.rmse_st);
				System.out.println("SAIW = " + -1.0 + ", AIW = " + -1.0 + ", ICR = " + -1.0 + "%");
				System.out.println("APT = " + each.apt + " msec" + ", AUT = " + -1.0 + " msec" + ", ATT = " + -1.0 + " msec");
				System.out.println("HPT = " + each.hpt + " msec" + ", HUT = " + -1.0 + " msec" + ", HTT = " + -1.0 + " msec");
				System.out.println("TPT = " + each.tpt + " msec" + ", TUT = " + -1.0 + " msec" + ", TTT = " + -1.0 + " msec");
				System.out.println("PTR = " + each.ptr + "%" + ", UTR = " + -1.0 + "%" + ", TTR = " + -1.0 + "%");
				System.out.println("TC = " + -1.0);
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
