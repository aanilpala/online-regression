package regression.online.test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import regression.online.learner.BayesianMAP;
import regression.online.learner.BayesianMAPForgetting;
import regression.online.learner.BayesianMAPForgetting;
import regression.online.learner.BayesianMLE;
import regression.online.learner.BayesianMLEForgetting;
import regression.online.learner.BayesianPredictive;
import regression.online.learner.BayesianMAPWindowed;
import regression.online.learner.BayesianMLEWindowed;
import regression.online.learner.BayesianPredictiveWindowed;
import regression.online.learner.NadaryaWatsonEstimator;
import regression.online.learner.Regressor;
import regression.online.model.Prediction;



public class RegressorTest {

	public Regressor reg;
	public List<double[][]> data_points;
	public List<Double> responses;
	public double rmse;
	public double interval_containment_rate;
	
	public RegressorTest(Regressor reg, List<double[][]> data_points, List<Double> responses) {
		
		if(data_points.size() != responses.size()) {
			System.out.println("Size Inconsistency");
			return;
		}
		
		this.data_points = data_points;
		this.responses = responses;
		this.reg = reg;
		this.rmse = 0;
		
	}
	
	public void test(boolean dump_logs) {
		int containment_count = 0;
		long accumulated_squared_error = 0;
		
		FileWriter fw = null;
		if(dump_logs) {
			try {
				fw = new FileWriter("/Users/anilpa/Desktop/tb_output/" + this.reg.name + "_logs.txt");
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		for(int ctr = 0; ctr < data_points.size(); ctr++) {
			double[][] dp = data_points.get(ctr);
			Double response = responses.get(ctr);
			
			Prediction pred = reg.predict(dp);
			reg.update(dp, response, pred);
			
			if(ctr > 5) {
				accumulated_squared_error += Math.pow((pred.point_prediction - response)*100000, 2);
				if(pred.upper_bound >= response && response >= pred.lower_bound) containment_count++;
			}
			
			if(dump_logs) {
				try {
					fw.write("target: " + response + " predicted: " + pred.toString());
					fw.write("\n");
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		
		rmse = Math.sqrt(accumulated_squared_error/(100000.0*100000.0));
		interval_containment_rate = containment_count / (double) (data_points.size() - 5);
		
		try {
			fw.write("SUMMARY\n");
			fw.write("RMSE = " + rmse + " Interval Containment Ratio = " + interval_containment_rate);
			fw.flush();
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	public double getRMSE(){
		return rmse;
	}
	
	public double getContainmentRate(){
		return interval_containment_rate;
	}
	
	public String getRegName() {
		return reg.name;
	}
	
	public static void main(String[] args) {
		
		List<double[][]> data_points = new ArrayList<double[][]>();
		List<Double> responses = new ArrayList<Double>();
		
		List<Regressor> regs = new ArrayList<Regressor>();
		List<RegressorTest> reg_testers = new ArrayList<RegressorTest>();
 		
		try(BufferedReader br = new BufferedReader(new FileReader("/Users/anilpa/Desktop/opdata/lfakeOpData.csv"))) {
		    for(String line; (line = br.readLine()) != null; ) {
		    	
		    	String[] tokens = line.split("\\|");
		    	String[] features_tokens = tokens[0].split("\t");
		    	
		    	double[][] dp = new double[features_tokens.length][1];
		    	
		    	for(int ctr = 0; ctr < features_tokens.length; ctr++) {
		    		dp[ctr][0] = Double.parseDouble(features_tokens[ctr].trim());
		    	}
		    	
		    	data_points.add(dp);
		    	
		    	String[] response_tokens = tokens[1].split("\t");
		    	double response = Double.parseDouble(response_tokens[response_tokens.length-1].trim())/1000.0; // in ms
		    	responses.add(response);
		    	
		    }
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		int input_width = data_points.get(0).length;
		
//		regs.add(new BayesianMLE(input_width, false));
//		regs.add(new BayesianMLE(input_width, true));
//		regs.add(new BayesianMLEWindowed(input_width, false));
//		regs.add(new BayesianMLEWindowed(input_width, true));
//		regs.add(new BayesianMLEForgetting(input_width, false));
//		regs.add(new BayesianMLEForgetting(input_width, true));
//		
//		regs.add(new BayesianMAP(input_width, false, 15, 10));
//		regs.add(new BayesianMAP(input_width, true, 15, 10));
//		regs.add(new BayesianMAPWindowed(input_width, false, 15, 10));
//		regs.add(new BayesianMAPWindowed(input_width, true, 15, 10));
		regs.add(new BayesianMAPForgetting(input_width, false, 15, 10));
		regs.add(new BayesianMAPForgetting(input_width, true, 15, 10));
//		
//		regs.add(new BayesianPredictive(input_width, false, 15, 10));
//		regs.add(new BayesianPredictive(input_width, true, 15, 10));
//		regs.add(new BayesianPredictiveWindowed(input_width, false, 15, 10));
//		regs.add(new BayesianPredictiveWindowed(input_width, true, 15, 10));
//		
//		regs.add(new NadaryaWatsonEstimator(input_width));
		
		for(Regressor each : regs)
			reg_testers.add(new RegressorTest(each, data_points, responses));

		for(RegressorTest each : reg_testers) {
			each.test(true);
			System.out.println(each.getRegName() + " RMSE= " + each.getRMSE() + " IContainment: " + each.getContainmentRate());
		}
		
	}
}
