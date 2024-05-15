%%

clc
clear all




%% I. Define paths

% jump to the current directory
cd(fileparts(matlab.desktop.editor.getActiveFilename));

pathdatabase = "C:\Users\basdu83\OneDrive - Linköpings universitet\_Teams\touch comm MNG Kinect\data_contact_2022\contact_IFF_per_unit\new_axes";


acquisition_folders = dir(pathdatabase);
acquisitions_foldername = {acquisition_folders.name}';
acquisitions_foldername = acquisitions_foldername(3:end); % remove . and ..
acq_idx = listdlg('PromptString',{'Select scan folders to be computed.',''},...
                  'ListString', acquisitions_foldername,...
                  'InitialValue', 1:length(acquisitions_foldername),...
                  'ListSize', [400 500]);
acquisitions_foldername = acquisitions_foldername(acq_idx);


nacq = length(acquisitions_foldername);

pathdatas = cell(nacq, 1);

for n = 1:nacq
  pathdatas{n} = fullfile(pathdatabase, acquisitions_foldername{n});
end

fprintf("I. Done.\n");



%% II. Extract data

data = cell(nacq, 1);
metadata = cell(nacq, 1);

show = true;

for n = 1:nacq
  fprintf("data file (%i/%i): %s\n", n, nacq, acquisitions_foldername{n});
  opts = detectImportOptions(pathdata);
  opts.SelectedVariableNames = ["velLat", "velLong", "velVert"];
  data{n} = readtable(pathdatas{n}, opts);

  opts.SelectedVariableNames = ["stimulus", "vel", "finger", "force"];
  opts.DataLines = [3 3];
  metadata{n} = readtable(pathdatas{n}, opts);

  if show
    plot(data{n}{:,:});
    fprintf("press a button to continue.\n");
    waitforbuttonpress;
  end
  fprintf("Acquisition done.\n");
end

fprintf("II. Done.\n");






%% III. shows path in 3D


limit_npoints = true;
npoints_max = 1000;


for n = 1:nacq
  
  T = data{n};
  md = metadata{n};
  fprintf("\n-------\n");
  fprintf("data file (%i/%i): %s\n", n, nacq, acquisitions_foldername{n});
  fprintf("Type = %s\n", md.stimulus{1})
  fprintf("Velocity = %i cm/sec\n", md.vel)
  fprintf("Size = %s\n", md.finger{1})
  fprintf("Force = %s\n", md.force{1})

  % remove the zeros in common for all axes (no contact time)
  X_ids = find(T{:,1});
  Y_ids = find(T{:,2});
  Z_ids = find(T{:,3});
  ids = intersect(intersect (X_ids, Y_ids), Z_ids);
  T = T(ids,:);
  
  
  mins = min(T{:,:});
  maxs = max(T{:,:});
  set(gca, 'Xlim', [mins(1) maxs(1)], 'Ylim', [mins(2) maxs(2)], 'Zlim', [mins(3) maxs(3)])
  
  if limit_npoints && npoints_max < size(T, 1)
    npoints = npoints_max;
  else
    npoints = size(T, 1);
  end
  poi = floor(linspace(1, size(T, 1), npoints)); % points of interest
  
  % Define a colormap
  cmap = jet(npoints);
  
  hold on
  p_start = [T{1,1}, T{1,2}, T{1,3}];
  for p = 2:npoints
    p_stop = [T{poi(p),1}, T{poi(p),2}, T{poi(p),3}];
    coord = [p_start; p_stop];
    l = line(coord(:,1), coord(:,2), coord(:,3), 'color', cmap(p, :));
    p_start = p_stop;
  end
  hold off

  title(sprintf("%s_%icm/sec_%s_%s", md.stimulus{1}, md.vel, md.finger{1}, md.force{1}), 'Interpreter', 'None')

  input("Press enter in the command line to continue.")
  delete(l);
end



