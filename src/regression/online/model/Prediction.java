package regression.online.model;

public class Prediction {

	public double point_prediction;
	public double upper_bound;
	public double lower_bound;
	
	public double opt1, opt2;
	
	public Prediction(double point_prediction, double upper_bound, double lower_bound) {
		this.point_prediction = point_prediction;
		this.upper_bound = upper_bound;
		this.lower_bound = lower_bound;
	}
	
	public Prediction() {
		this.point_prediction = Double.NaN;
		this.upper_bound = Double.NaN;
		this.lower_bound = Double.NaN;
	}

	public Prediction(double pp, double up, double lo, double opt1, double opt2) {
		this.point_prediction = pp;
		this.upper_bound = up;
		this.lower_bound = lo;
		this.opt1 = opt1;
		this.opt2 = opt2;
	}

	public Prediction(double y, double predictive_deviance) throws Exception {
		if(predictive_deviance < 0) throw new Exception("NEGATIVE PREDICTIVE VARIANCE! THIS SHOULD NOT HAPPEN!");
		
		this.point_prediction = y;
		this.lower_bound = y - 1.96*predictive_deviance;
		this.upper_bound = y + 1.96*predictive_deviance;
		
	}

	@Override
	public String toString() {
		return point_prediction + " ∈ (" + ((Double) lower_bound).toString() + ", " + ((Double) upper_bound).toString() + ")";  
	}
}
