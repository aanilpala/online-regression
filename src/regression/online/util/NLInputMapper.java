package regression.online.util;

public class NLInputMapper {

	public int exp;
	public int input_dim;
	public int feature_dim;
	public int[][] input_mapping;
	public int log_map_size;
	
	public NLInputMapper(int input_dim, int exp, boolean log_features_enabled) {
		
		if(log_features_enabled) log_map_size = input_dim;
		else log_map_size = 0;
		
		if(exp > 3) {
			System.out.println("Generic Mapper is not supported yet");
			return;
		}
		
		// compute number of features
		
		this.exp = exp;
		feature_dim = 0;
		for(int ctr = 1; ctr <= exp; ctr++) {
			int nom = 1;
			int denom = 1;
			for(int ctr2 = ctr; ctr2 > 0; ctr2--) {
				denom *= ctr2;
				nom *= input_dim + ctr - 1 - (ctr - ctr2);
			}
			feature_dim += nom/denom;
		}
		
		feature_dim += log_map_size;
		input_mapping = new int[feature_dim][exp];
		
		for(int ctr = 0; ctr < feature_dim; ctr++) {
			for(int ctr2 = 0; ctr2 < exp; ctr2++) {
				input_mapping[ctr][ctr2] = Integer.MIN_VALUE; // means non-existent/invalid
			}
		}
		
		int map_counter = 0;
		
		for(int ctr = 0; ctr < input_dim; ctr++) {
			input_mapping[map_counter++][0] = ctr;
		}
		
		
		for(int ctr = 0; exp >= 2 && ctr < input_dim; ctr++) {
			for(int ctr2 = ctr; ctr2 < input_dim; ctr2++) {
				input_mapping[map_counter][0] = ctr;
				input_mapping[map_counter++][1] = ctr2;
			}
		}
		
		for(int ctr = 0; exp >= 3 && ctr < input_dim; ctr++) {
			for(int ctr2 = ctr; ctr2 < input_dim; ctr2++) {
				for(int ctr3 = ctr2; ctr3 < input_dim; ctr3++) {
					input_mapping[map_counter][0] = ctr;
					input_mapping[map_counter][1] = ctr2;
					input_mapping[map_counter++][2] = ctr3;
				}
			}
		}
		
		for(int ctr = 0; log_features_enabled &&ctr < input_dim; ctr++) {
			input_mapping[map_counter++][0] = -1*(ctr+1);
		}	
	}
	
	public double[][] map(double[][] input) {
		
		// adding log variants
		//feature_dim += input_dim;
		
		double[][] features = new double[feature_dim][1];
		
		for(int ctr = 0; ctr < feature_dim; ctr++) {
			features[ctr][0] = 1.0f;
			for(int ctr2 = 0; ctr2 < exp; ctr2++) {
				if(input_mapping[ctr][ctr2] >= 0) features[ctr][0] *= input[input_mapping[ctr][ctr2]][0];
				else if(input_mapping[ctr][ctr2] > Integer.MIN_VALUE) features[ctr][0] *= Math.log(input[-1*input_mapping[ctr][ctr2]-1][0]);
			}
		}
		
//		for(int ctr = feature_dim; ctr > feature_dim - input_dim; ctr--) {
//			features[ctr-1][0] = Math.log(input[ctr-(feature_dim-input_dim)][0]);
//		}
		
		return features;
	}
	
}
