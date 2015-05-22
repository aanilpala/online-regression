package regression.online.util;
import java.text.DecimalFormat;


public class MatrixPrinter {

	static DecimalFormat df = new DecimalFormat("#.######");

	public static void print_matrix(double[][] m) {
		
		for(int ctr = 0; ctr < m.length ; ctr++) {
			for(int ctr2 = 0; ctr2 < m[ctr].length; ctr2++) {
				System.out.print(df.format(m[ctr][ctr2]));
				System.out.print(" ");
			}
			System.out.println();
		}
		System.out.println();
		
	}
}
