// ......


var pmCmds = newMenu("Popup Menu",
	newArray("Help...", "Rename...", "Duplicate...", "Original Scale",
	"Paste Control...", "-", "Record...", "Capture Screen ", "Monitor Memory...",
	"Find Commands...", "Control Panel...", "Startup Macros...", "Search...","Set default ROI ...", "Set Scale to 20 um ...", "Save measurments...", "Predict coating..."));

macro "Popup Menu" {
	cmd = getArgument();
	if (cmd=="Help...")
		showMessage("About Popup Menu",
			"To customize this menu, edit the line that starts with\n\"var pmCmds\" in ImageJ/macros/StartupMacros.txt.");


	else if (cmd == "Set default ROI ..."){
		imageTitle=getTitle();
		imageDir=getDirectory("image");
		imagePath=imageDir+imageTitle;

        // Find the position of the last dot in the filename
		lastDotIndex = lastIndexOf(imageTitle, ".");
		
		if (lastDotIndex != -1) {
		    // Extract the part before the last dot
		    imageTitle2 = substring(imageTitle, 0, lastDotIndex);
		} else {
		    imageTitle2 = imageTitle;
		}
		dir_scripts = getDirectory("macros");
		scriptpath= dir_scripts  + "Folder_name_placeholder" + File.separator + "create_roi.py";
		exec("python", scriptpath, imagePath);   
        roiPath = imageDir+imageTitle2+"_line_rois.zip"; 	

    	if (File.exists(roiPath)) {
            run("ROI Manager...");
            // Get the ROI Manager instance
            roiManager("show all with labels");
            roiManager("Open", roiPath);
        } else {
            showMessage("Error: ROI file not found. Make sure you performed the 'Set default ROI ...' operation correctly.Set default ROI ... is used for images only.");
        }
 
	}
		
	else if (cmd == "Set Scale to 20 um ..."){
		setScale(0.07881773, 0.07881773, "um"); // Set scale to 20 micrometers per 253.75 pixels
		}


	else if (cmd == "Predict coating...") {
		 // change based on how many rois lines you want
		 num_of_roi_lines = 10;

   		 python_path = "python";
   		 dir_scripts = getDirectory("macros");
   		 script_path = dir_scripts  + "Folder_name_placeholder" + File.separator + "api.py"; // Use File.separator for platform-independent file separators
  		 imageTitle = getTitle();
    	 imageDir = getDirectory("image");
    	 
   		 input_image_path = imageDir + imageTitle;

   		 // Build command to execute Python script
 		 command = python_path + " " + script_path + " " + input_image_path + " " + num_of_roi_lines; 
  		 roi_path = exec(command);
  		 length = lengthOf(roi_path);
   		 roi_path_without_whitespace = substring(roi_path, 0, length - 1);
   		 
   		if (File.exists(roi_path_without_whitespace)) {
            run("ROI Manager...");
            // Get the ROI Manager instance
            roiManager("show all with labels");
            roiManager("Open", roi_path_without_whitespace);
        } else {
            showMessage("Error: ROI file not found. Make sure you performed the 'Predict coating...' operation correctly.Predict is used for images only.");
        }
    
		}

	else if (cmd == "Save measurments..."){
		imageTitle=getTitle();
		imageDir=getDirectory("image");
		imagePath=imageDir+imageTitle;
		lastDotIndex = lastIndexOf(imageTitle, ".");
		
		if (lastDotIndex != -1) {
		    // Extract the part before the last dot
		    imageTitle2 = substring(imageTitle, 0, lastDotIndex);
		} 
		else {
		    imageTitle2 = imageTitle; // No change if there is no dot
		}
		
		roiManager("Save", imageDir+imageTitle2+"_measurements.zip"); // Save measurements
		}
	}
	else
		run(cmd);	



