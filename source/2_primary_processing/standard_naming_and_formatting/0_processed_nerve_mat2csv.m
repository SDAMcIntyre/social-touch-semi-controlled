

% Specify the folder where the .mat files are located
database_path = ['E:\OneDrive - Linköpings universitet\_Teams\touch comm MNG Kinect\'...
                 'basil_tmp\data\semi-controlled'];
input_dir = 'processed\nerve\0_matlab_files_corrected-date';
input_dir_abs = fullfile(database_path, input_dir);

output_dir = 'processed\nerve\1_csv_files';
output_dir_abs = fullfile(database_path, output_dir);
if ~exist(output_dir_abs, 'dir')
    mkdir(output_dir_abs);
    disp(['Folder "', output_dir_abs, '" created successfully.']);
else
    disp(['Folder "', output_dir_abs, '" already exists.']);
end

% Get a list of all .mat files in the folder
matFiles = dir(fullfile(input_dir_abs, '*.mat'));

% Loop through each .mat file
for k = 1:length(matFiles)
    % Construct the full file name
    matFileName = fullfile(input_dir_abs, matFiles(k).name);
    
    % Load the .mat file
    data = load(matFileName);
    
    % Extract the structure S
    S = data.S;
    
    % Extract metadata attributes from S
    Exp = S.Exp;
    Unit = S.Unit;
    Zoom = S.Zoom;
    UnitName = S.UnitName;
    UnitNumber = S.UnitNumber;
    IdxInDataInfo = S.IdxInDataInfo;
    UnitTyp = S.UnitType;
    Stimulus = S.Stimulus;

    if Stimulus ~= "Semi_contr"
      continue
    end
    
    fprintf('File: %s\n', matFiles(k).name);
    fprintf('Exp: %i\n', Exp);
    fprintf('Unit: %i\n', Unit);
    fprintf('Zoom: %i\n', Zoom);
    fprintf('UnitName: %s\n', UnitName);
    fprintf('UnitNumber: %i\n', UnitNumber);
    fprintf('IdxInDataInfo: %i\n', IdxInDataInfo);
    fprintf('UnitTyp: %s\n', UnitTyp);
    fprintf('Stimulus: %s\n', Stimulus);
    fprintf('\n'); % Add a blank line for separation

    % create neuron folder's name
    d = S.FullPeriod_D.ContD.D;
    dateStr = num2str(d.YYYYMMDD(1));
    % Extract year, month, and day parts
    yearPart = dateStr(1:4);
    monthPart = dateStr(5:6);
    dayPart = dateStr(7:8);
    % Concatenate parts
    foldername = [yearPart '-' monthPart '-' dayPart '_' UnitName];
    current_folder_abs = fullfile(output_dir_abs, foldername);
    if ~exist(current_folder_abs, 'dir')
        mkdir(current_folder_abs);
        disp(['Folder "', current_folder_abs, '" created successfully.']);
    else
        disp(['Folder "', current_folder_abs, '" already exists.']);
    end

    % Construct the metadata text file name
    [~, name, ~] = fileparts(matFiles(k).name);
    metadataFileName = fullfile(current_folder_abs, [name, '_metadata.txt']);
    % Open the metadata text file for writing
    metadataFile = fopen(metadataFileName, 'w');
    
    % Write the metadata to the text file
    fprintf(metadataFile, 'File: %s\n', matFiles(k).name);
    fprintf(metadataFile, 'Exp: %i\n', Exp);
    fprintf(metadataFile, 'Unit: %i\n', Unit);
    fprintf(metadataFile, 'Zoom: %i\n', Zoom);
    fprintf(metadataFile, 'UnitName: %s\n', UnitName);
    fprintf(metadataFile, 'UnitNumber: %i\n', UnitNumber);
    fprintf(metadataFile, 'IdxInDataInfo: %i\n', IdxInDataInfo);
    fprintf(metadataFile, 'UnitTyp: %s\n', UnitTyp);
    fprintf(metadataFile, 'Stimulus: %s\n', Stimulus);
    fprintf(metadataFile, '\n'); % Add a blank line for separation
    
    % Close the metadata text file
    fclose(metadataFile);
    
    data = S.FullPeriod_D.ContD;
    nblocks = length(data);
    for b = 1:nblocks
      % Extract the table FullPeriod_D.ContD.D
      tableData = data(b).D;
      blockid_str = sprintf('block%01d', b);
      % Construct the CSV file name
      csvFileName = fullfile(current_folder_abs, [name '_' blockid_str '_table.csv']);
      
      % Save the table to a CSV file
      writetable(tableData, csvFileName);
    end
end

disp('Metadata and tables have been saved successfully.');












