package regression.online.learner;
import regression.online.model.Prediction;
import regression.online.util.NLInputMapper;


public abstract class Regressor {

	public int feature_count;
	NLInputMapper nlinmap;
	public boolean map2fs;
	public String name;
	
	public Regressor(boolean map2fs2, int input_width) {
		
		this.map2fs = map2fs2;
		
		if(map2fs) {
			nlinmap = new NLInputMapper(input_width, 2, true);
			this.feature_count = nlinmap.feature_dim;
		}
		else {
			this.feature_count = input_width;
		}
	}
	public Prediction predict(double[][] dp) { return null; };
	public void update(double[][] dp, double y, Prediction prediction) {};
	
}
