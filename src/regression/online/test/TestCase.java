package regression.online.test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TestCase {

	List<double[][]> data_points;
	List<Double> responses;
	int input_width;
	String data_set_name;
	
	public TestCase(File file) throws Exception {
		data_points = new ArrayList<double[][]>();
		responses = new ArrayList<Double>();
		
		BufferedReader br = new BufferedReader(new FileReader(file));
		for(String line; (line = br.readLine()) != null; ) {
	    	
	    	String[] tokens = line.split("\\|");
	    	String[] features_tokens = tokens[0].split("\t");
	    	
	    	double[][] dp = new double[features_tokens.length][1];
	    	
	    	for(int ctr = 0; ctr < features_tokens.length; ctr++)
	    		dp[ctr][0] = Double.parseDouble(features_tokens[ctr].trim());
	    	
	    	data_points.add(dp);
	    	
	    	String[] response_tokens = tokens[1].split("\t");
	    	double response = Double.parseDouble(response_tokens[response_tokens.length-1].trim()); // in ms
	    	responses.add(response);
		}
		
		data_set_name = file.getName().split("\\.")[0];
		input_width = data_points.get(0).length;
	}
}