
% To execute before 0_processed_nerve_mat2csv.m


% Specify the folder where the .mat files are located
database_path = ['E:\OneDrive - Linköpings universitet\_Teams\touch comm MNG Kinect\'...
                 'basil_tmp\data\semi-controlled'];
input_dir = 'processed\nerve\0_matlab_files';
input_dir_abs = fullfile(database_path, input_dir);

output_dir = 'processed\nerve\0_matlab_files_corrected-date';
output_dir_abs = fullfile(database_path, output_dir);
if ~exist(output_dir_abs, 'dir')
    mkdir(output_dir_abs);
    disp(['Folder "', output_dir_abs, '" created successfully.']);
else
    disp(['Folder "', output_dir_abs, '" already exists.']);
end

% Get a list of all .mat files in the folder
matFiles = dir(fullfile(input_dir_abs, '*.mat'));

% define the list of sessions of interest
matFilesOfInterest = {'ST14-01', 'ST14-02', 'ST16-01', 'ST16-02'};

% Loop through each .mat file
for k = 1:length(matFiles)
  
  % Construct the full file name
  matFileName = fullfile(input_dir_abs, matFiles(k).name);
  
  % Load the .mat file
  load(matFileName);
  
  % modify the files of interest
  if contains(matFiles(k).name, matFilesOfInterest)
    data = S.FullPeriod_D.ContD;
    nblocks = length(data);
    for b = 1:nblocks
      % Access the table in the structure
      dataTable = data(b).D;
      
      % Assuming the date column is named 'Date' and is in YYYYMMDD format
      % Convert the date to datetime format
      dataTable.YYYYMMDD = datetime(num2str(dataTable.YYYYMMDD), 'InputFormat', 'yyyyMMdd');
      
      % Subtract one day from the date
      if contains(matFiles(k).name, 'ST16-02')
        newdate = dataTable.YYYYMMDD + days(2);
      else
        newdate = dataTable.YYYYMMDD - days(1);
      end
      
      % Convert the date back to YYYYMMDD format
      dataTable.YYYYMMDD = str2double(cellstr(datestr(newdate, 'yyyymmdd')));
      
      % Update the table in the structure
      S.FullPeriod_D.ContD(b).D = dataTable;
    end

    matFileName = fullfile(input_dir_abs, matFiles(k).name);
    outputFileName = fullfile(output_dir_abs, matFiles(k).name);
    % Save the updated structure back to the .mat file
    save(outputFileName, 'S');  % Replace 'yourfile.mat' with your actual .mat file name
  end
  
end

disp('Matlab file have been saved successfully.');












