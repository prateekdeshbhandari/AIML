import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import java.io.IOException;

public class MaxTemperature {

    // Mapper class
    public static class TemperatureMapper extends Mapper<Object, Text, Text, IntWritable> {
        private Text year = new Text();
        private IntWritable temperature = new IntWritable();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            // Split the input line by comma
            String[] parts = value.toString().split(",");
            
            if (parts.length == 2) {
                try {
                    // Extract year and temperature
                    year.set(parts[0].trim());
                    temperature.set(Integer.parseInt(parts[1].trim()));
                    
                    // Emit the year and temperature as key-value pair
                    context.write(year, temperature);
                } catch (NumberFormatException e) {
                    // Ignore the record if it has an invalid temperature
                }
            }
        }
    }

    // Reducer class
    public static class TemperatureReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable maxTemperature = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int maxTemp = Integer.MIN_VALUE;

            // Iterate over all temperature values for a given year and find the max
            for (IntWritable val : values) {
                maxTemp = Math.max(maxTemp, val.get());
            }

            maxTemperature.set(maxTemp);
            
            // Emit the year and the maximum temperature
            context.write(key, maxTemperature);
        }
    }

    // Main Driver class
    public static void main(String[] args) throws Exception {
        // Initialize configuration
        Configuration conf = new Configuration();
        GenericOptionsParser parser = new GenericOptionsParser(conf, args);
        args = parser.getRemainingArgs();

        if (args.length != 2) {
            System.err.println("Usage: MaxTemperature <input> <output>");
            System.exit(2);
        }

        // Create a new MapReduce job
        Job job = Job.getInstance(conf, "Max Temperature");
        job.setJarByClass(MaxTemperature.class);

        // Set Mapper and Reducer classes
        job.setMapperClass(TemperatureMapper.class);
        job.setReducerClass(TemperatureReducer.class);

        // Set the output key and value types
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        // Set input and output paths
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        // Wait for the job to finish
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}

