package com.sakha.taskscheduler.CrontabTask;

import java.io.File;
import java.io.IOException;
import java.time.Instant;
import java.util.Calendar;
import java.util.Date;

/**
 * Hello world!
 *
 */
public class App {
	public static void main(String[] args) {
		System.out.println("Hello World!");
		try {

			Calendar cal = Calendar.getInstance();
			cal.setTime(Date.from(Instant.now()));

			String filename = String.format("file-%1$tY-%1$tm-%1$td-%1$tk-%1$tM-%1$tS-%1$tp.txt", cal);

			String folderToSave = "/home/shubham-sakha/Desktop/DataScience";

			File file = new File(folderToSave, filename);

			System.out.println("Final filepath : " + file.getAbsolutePath());
			if (file.createNewFile()) {
				System.out.println("File is created!");
			} else {
				System.out.println("File is already existed!");
			}

		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
