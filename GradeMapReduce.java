import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import java.io.IOException;

public class GradeMapReduce {

    // Function to convert numeric grade to letter grade
    public static String mapGrades(int grade) {
        if (grade >= 90) {
            return "A";
        } else if (grade >= 80) {
            return "B";
        } else if (grade >= 70) {
            return "C";
        } else if (grade >= 60) {
            return "D";
        } else {
            return "F";
        }
    }

    public static class GradeMapper extends Mapper<Object, Text, Text, IntWritable> {

        private Text studentName = new Text();
        private IntWritable grade = new IntWritable();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] parts = line.split("\\s+");
            if (parts.length == 2) {
                studentName.set(parts[0]);
                grade.set(Integer.parseInt(parts[1]));
                context.write(studentName, grade);
            }
        }
    }

    public static class GradeReducer extends Reducer<Text, IntWritable, Text, Text> {

        private Text result = new Text();

        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            int count = 0;

            // Accumulate all grades for the student
            for (IntWritable val : values) {
                sum += val.get();
                count++;
            }

            // Calculate the average grade
            int average = sum / count;
            String letterGrade = mapGrades(average);

            // Output student and their letter grade
            result.set(letterGrade);
            context.write(key, result);
        }
    }

    
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "GradeMapReduce");
        job.setJarByClass(GradeMapReduce.class);
        job.setMapperClass(GradeMapper.class);
        job.setReducerClass(GradeReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job,new Path(args[0]));
        FileOutputFormat.setOutputPath(job,new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}

