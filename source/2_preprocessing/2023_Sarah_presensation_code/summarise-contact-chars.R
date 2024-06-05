#### libraries ####
library(readr)
library(dplyr)
library(stringr)
library(ggplot2)
library(patchwork)
library(Hmisc)
library(tidyr)

#### data folders ####
DATA_FOLDER <- "~/Library/CloudStorage/OneDrive-Linköpingsuniversitet/projects - in progress/touch comm MNG Kinect/"
CONTACT_DATA_FOLDER <- paste0(DATA_FOLDER, "data_contact-IFF-trial/with_contact_flag/")
STIM_INFO_FOLDER <- paste0(DATA_FOLDER, "data_stimulus-logs/")

# notes from Shan about the data:
# the csv file contains the contact features as 
# raw data (named as xxRaw), interpolated data (named as xx), smoothed data (named as xxSmooth), 
# 1st derivative (named as xx1D), and 2nd derivative (named as xx2D). 
# The contact data were upsampled and the neural data were downsampled to both 1000Hz.
# the units for first and second derivatives are 
# cm/s2 for velocity1D, cm/s for depth1D, and cm2/s for area1D 
# I  kept the data of the whole trial regardless of having neural data or not. 
# The end of each trial might not be covered by the trial number since the contact sometimes stopped 
# after the LED went off.

#### plot appearance ####
theme_set(theme_bw(base_size = 14))

#### plot a single session ####

ex <- read_csv(paste0(CONTACT_DATA_FOLDER, "2022-06-17/unit5/2022-06-17_18-15-56_controlled-touch-MNG_ST16_5_block1.csv"))

plot_feature <- function(df, feature, y_axis_label, trial_flag) {
  df %>% 
    mutate(Stimulus = as.character(.data[[trial_flag]])) %>% 
    ggplot(aes(x = t, y = .data[[feature]], colour = Stimulus)) +
    geom_point(size = 0.2) +
    labs(y = y_axis_label) + 
   theme(legend.position = "bottom")
}

plot_session <- function(df, trial_flag = "trial_id", title = "") {
  
  theme_no_x <- theme(
    axis.title.x=element_blank(),
    axis.text.x=element_blank(),
    axis.ticks.x=element_blank())
  theme_no_legend <- theme(legend.position = "none")
  
  area <- plot_feature(df, "areaSmooth", expression("Contact cm"^2), trial_flag) + theme_no_x + theme_no_legend
  depth <- plot_feature(df, "depthSmooth", "Depth cm", trial_flag) + theme_no_x + theme_no_legend
  velAbs <- plot_feature(df, "velAbsSmooth", "AbsV cm/s", trial_flag) + theme_no_x + theme_no_legend
  velLat <- plot_feature(df, "velLatSmooth", "LatV cm/s", trial_flag) + theme_no_x + theme_no_legend
  velLong <- plot_feature(df, "velLongSmooth", "LongV cm/s", trial_flag) + theme_no_x + theme_no_legend
  velVert <- plot_feature(df, "velVertSmooth", "VertV cm/s", trial_flag) + theme_no_x + theme_no_legend
  
  iff <- df %>% 
    mutate(
      spike_label = if_else(spike == 1, "|", ""),
      Stimulus = as.character(.data[[trial_flag]])
    ) %>% 
    ggplot(aes(x = t, y = IFF, colour = Stimulus)) +
    geom_line(linewidth = 0.2) +
    geom_text(aes(y = -max(IFF)/5, label = spike_label), alpha = 0.5, size = 8, show.legend = FALSE) +
    labs(y = "IFF Hz", x = "Seconds") +
    theme(legend.position = "bottom")
  
  area / depth / velAbs / velLat / velLong / velVert / iff + 
    #plot_layout(guides = 'collect') +
    #theme(legend.position = "bottom") +
    plot_annotation(title = title)
}

plot_session(ex)


#### read in semi-controlled data ####

data_files_controlled <- list.files(CONTACT_DATA_FOLDER, "controlled", full.names = TRUE, recursive = TRUE)

stim_files_controlled <- list.files(STIM_INFO_FOLDER, "stimuli", full.names = TRUE, recursive = TRUE)

merge_session_data_w_stiminfo <- function(data_file_list, stim_file_list) {
  
  # read in the stim files
  read_all_stim_files <- function(file_list) {
    stim_file_contents <- tibble()
    for (stimfile in stim_files_controlled) {
      stim_file_contents <- rbind(
        stim_file_contents,
        read_csv(stimfile, show_col_types = FALSE)
      )
    }
    stim_file_contents %>% 
      mutate(kinect_recording = basename(str_replace_all(kinect_recording, "\\\\", "/")) )
  }
  
  stimuli_controlled <- read_all_stim_files(stim_file_list)
  
  # read in the data files and match with stim files
  data_controlled <- tibble()
  for (f in data_files_controlled) {
    print(f)
    fname <- basename(f)
    
    # match based on date/time stamp to find stim info for this session
    session_datetime <- str_extract(fname, "([0-9]|-|_){19}")
    stim_idx <- which(str_detect(stimuli_controlled$kinect_recording, session_datetime))
    session_stim <- stimuli_controlled[stim_idx,]
    
    # read data
    session_data <- read_csv(f, show_col_types = FALSE) 
    
    # get the trial_ids based on the LED visible on the camera
    trial_ids <- na.omit(unique(session_data$trial_id))
    
    # check if the stim file has the same number of trials
    if (nrow(session_stim) == length(trial_ids)) {
      
      # create the trial variable in the session data file to later merge with the stim info
      session_data <- session_data %>% 
        mutate(trial = NA_integer_) 
      
      # fill the new trial variable with labels from the session stim info
      for (trial_n in seq_along(trial_ids)) {
        session_data$trial[session_data$trial_id == trial_ids[trial_n]] <- session_stim$trial[trial_n]
      }
      # merge session data with stim info
      session_data <- full_join(session_data, session_stim, by = "trial") %>% 
        #  add filename metadata and unique stim description to new variables 
        mutate(
        filename = fname,
        unit = str_extract(fname, "ST[0-9]+_[0-9]+"),
        stim_desc = if_else( is.na(trial_id), NA_character_,
          paste0(
          str_pad(trial_id, 2, pad = "0"), ".",
          " ", type, 
          " speed", str_pad(speed, 2, pad ="0"), 
          " ", str_extract(contact_area, "(finger)|(hand)"),
          " ", force )
      ))
      
    } else {
      warning("number of stimuli does not match")
      print("number of stimuli does not match")
    }
    
    # update to all data 
    data_controlled <- rbind(data_controlled, session_data )
  }
  data_controlled
}

data_controlled <- merge_session_data_w_stiminfo(data_files_controlled, stim_files_controlled)

#### fix stimulus alignment ####

estimate_experimenter_lag <- function(contact, led) {
  contact_flag <- if_else(contact == 0, 0, 1)
  led_flag <- if_else(is.na(led), 0, 1)
  
  cc_contact_led <- ccf(contact_flag, led_flag, lag.max = 2000, plot = FALSE)
  
  list(
    cc_plot = tibble(lag = cc_contact_led$lag, cc = cc_contact_led$acf) %>% 
      ggplot(aes(x = lag, y = cc)) +
      geom_point(),
    lag_estimate = cc_contact_led$lag[which(cc_contact_led$acf == max(cc_contact_led$acf))]
  ) 
}

result <- list()
for (session_n in seq_along(unique(data_controlled$filename))) {
# for (session_n in 1:2) {
# for (session_n in 12:length(unique(data_controlled$filename))) {
  session_fname <- unique(data_controlled$filename)[session_n]
  print(paste0(session_n, " of ", length(unique(data_controlled$filename)), ": ", session_fname ))

  result[[session_n]] <- list()
  result[[session_n]]$filename <- session_fname
  
  session_data <- data_controlled %>% filter(filename == session_fname)
  
  cc_result <- estimate_experimenter_lag(session_data$Contact_Flag, session_data$trial_id)
  lag_estimate <- cc_result$lag_estimate
  cc_plot <- cc_result$cc_plot + 
    labs(title = paste0("lag = ",lag_estimate)) +
    plot_annotation(caption = session_fname)
  
  if (lag_estimate > 0) {
    fill_value <- na.omit(session_data$stim_desc)[1]
    session_data <- session_data %>% 
      mutate(stim_desc_shifted = lag(stim_desc, lag_estimate, fill_value))
    
  } else {
    fill_value <- na.omit(session_data$stim_desc)[length(na.omit(session_data$stim_desc))]
    session_data <- session_data %>% 
      mutate(stim_desc_shifted = lead(stim_desc, -lag_estimate, fill_value))
    
    }
  
  # simple fill
  session_data <- session_data %>% 
    mutate(stim_desc_filled = stim_desc) %>% 
    tidyr::fill(stim_desc_filled, .direction = "downup") %>% 
    # also apply improvements to individual stim descriptors 
    mutate(
      type = str_extract(stim_desc_filled, "(tap)|(stroke)"),
      speed = str_extract(stim_desc_filled, "speed[0-9-]{2}") %>% str_replace("speed", ""),
      contact_area = str_extract(stim_desc_filled, "(finger)|(hand)"),
      force = str_extract(stim_desc_filled, "(light)|(moderate)|(strong)"),
      stim_desc_unordered = paste0(type," ", contact_area,
                                   " ", force, " speed", speed)
      )
  
  result[[session_n]]$session_data <- session_data
  
  result[[session_n]]$lag_estimate <- lag_estimate
  
  result[[session_n]]$plot_cc <- cc_plot

  result[[session_n]]$plot_before <- session_data %>%
    plot_session("stim_desc", paste(session_fname, "before"))

  result[[session_n]]$plot_after_cc <- session_data %>%
    plot_session("stim_desc_shifted", paste(session_fname, "after.cc"))
  
  result[[session_n]]$plot_after_sf <- session_data %>%
    plot_session("stim_desc_filled", paste(session_fname, "after.sf"))

}


####. save plots ####

plot_folder <- "figures_stim-align-cc-fill/"
for (r in seq_along(result)) {
# for (r in 1:2) {
  print(paste0(r, " of ", length(result), ": ", result[[r]]$filename ))
  
  # cross correlation
  print(result[[r]]$plot_cc)
  result[[r]]$filename %>% 
    str_replace("\\.csv", "_cc.png") %>% 
    paste0(plot_folder, .) %>% 
    ggsave(width = 8, height = 6)

  # before
  print(result[[r]]$plot_before)
  result[[r]]$filename %>% 
    str_replace("\\.csv", "_before.png") %>% 
    paste0(plot_folder, .) %>% 
    ggsave(width = 20, height = 12)

  #after
  print(result[[r]]$plot_after_cc)
  result[[r]]$filename %>% 
    str_replace("\\.csv", "_after-cc.png") %>% 
    paste0(plot_folder, .) %>% 
    ggsave(width = 20, height = 12)
  
  #after
  print(result[[r]]$plot_after_sf)
  result[[r]]$filename %>% 
    str_replace("\\.csv", "_after-sf.png") %>% 
    paste0(plot_folder, .) %>% 
    ggsave(width = 20, height = 12)

}

# re-combine corrected data
data_controlled <- tibble()
for (r in seq_along(result)) {
  data_controlled <- rbind(
    data_controlled, 
    result[[r]]$session_data
    )
}


#### dataset plots ####

file_list <- unique(data_controlled$filename)

for (session_number in 1:length(file_list)) {
  session_file_name <- file_list[session_number]
  print(paste0("session ", session_number, " of ", length(file_list), ": ", session_file_name))
  
    session_data <- data_controlled %>% 
      filter(filename == session_file_name)
    
    stim_list <- unique(session_data$stim_desc_unordered)
    
    for (stim_name in stim_list) {
      
      plot_folder <- paste0("figures_stim-plots/",stim_name,"/")
      if (!dir.exists(file.path(plot_folder)) ) dir.create(file.path(plot_folder), recursive = TRUE)
      
      # all sessions overlay
      
      overlay_plot_name <- paste0(plot_folder, "all_sessions_overlay.png")
      
      if (!file.exists(overlay_plot_name)) {
        overlay_plot <- data_controlled %>% 
          filter(stim_desc_unordered == stim_name) %>% 
          mutate(
            file_info = str_remove(filename, "controlled-touch-MNG_") %>% 
              str_remove("\\.csv")
          ) %>% 
          plot_session("file_info", stim_name)
        
        print(overlay_plot)
        overlay_plot_name %>% 
          ggsave(width = 20, height = 12)
      }
      

      # single session
      
      stim_data <- session_data %>% 
        filter(stim_desc_unordered == stim_name)
      
      single_plot_name <- session_file_name %>% 
        str_replace("\\.csv", 
                    paste0("_", str_extract(stim_data$stim_desc_filled[1], "[0-9]{2}\\."), "png")
        ) %>% 
        paste0(plot_folder, .)
      
      if (!file.exists(single_plot_name)) {
        stim_plot <- stim_data %>% 
          plot_session("stim_desc_unordered", stim_name) + plot_annotation(caption = session_file_name)
        
        print(stim_plot)
        single_plot_name %>% 
          ggsave(width = 20, height = 12)
        }
    }
}

#### overlay plots ####

# single stim
data_controlled %>% 
  mutate(
    file_info = str_remove(filename, "controlled-touch-MNG_") %>% 
      str_remove("\\.csv")
    ) %>% 
  filter(stim_desc_unordered == stim_name) %>% 
  ggplot(aes(x = t, y = areaSmooth)) +
  geom_line(aes(colour = file_info)) +
  labs(x = "Seconds", title = stim_name) +
  theme(legend.position = "bottom")

# Stroking with the hand
data_controlled %>% 
  filter(
      Contact_Flag != 0 &
      type == "stroke" &
      str_detect(contact_area, "hand")
    ) %>% 
  mutate(stim_label = paste0(force, " ", speed," cm/s" ) ) %>% 
  ggplot(aes(x = t, y = velLongSmooth, colour = force)) +
  facet_wrap(~ stim_label, scales = "free", ncol = 5) +
  geom_line(aes(group = filename), show.legend = FALSE, size = 0.5, alpha = 0.4) +
  scale_y_continuous(limits = c(-60,80)) +
  labs(
    title = "Stroking with the hand", 
    y = "Longitudinal velocity (cm/s)", 
    x = "Seconds"
    )


#### session plot ####
data_controlled %>% 
  filter(filename == "2022-06-17_17-48-40_controlled-touch-MNG_ST16_5_block4.csv") %>% 
  plot_session("stim_desc_unordered")

ggsave("figures/session.png", width = 20, height = 12)

#### summary data plots ####

data_controlled_summary <- data_controlled %>% 
  # filter time windows based on manual inspection of trace plots above
  # filter( !(stim_desc_filled == "tap finger light speed01" & t < 5) ) %>% 
  # tap finger light speed02 OK
  
  # tap finger light speeed09 
    # check "2022-06-17_18-08-29_controlled-touch-MNG_ST16_5_block1.csv"
  # filter( !filename == "2022-06-17_17-43-26_controlled-touch-MNG_ST16_5_block1.csv" & t < 31 ) %>% 
  
  # tap hand moderate speed01 # DRAW THIS ONE TO CHECK THE TIME
  # filter( !(filename == "2022-06-17_18-17-20_controlled-touch-MNG_ST16_5_block2.csv" & t > 50) ) %>% 
  
  # filter( !(stim_desc_filled == "tap hand moderate speed03" & t < 11.25 ) ) %>% 
  # tap hand light speed09 
  filter(Contact_Flag!=0) %>% 
  filter(areaSmooth > 0.2) %>% 
  group_by(stim_desc_unordered, type, contact_area, force, speed) %>% 
  summarise(across(
    .cols = ends_with("Smooth"),
    .fns = list(
      mean_dir = ~ mean(.),
      mean = ~ mean(abs(.)),
      med = ~ median(abs(.)),
      p75 = ~ quantile(abs(.), 0.75)
    ),
    .names = "{.col}_{.fn}" )
  )


theme_set(theme_bw(base_size = 14) + theme(strip.background = element_rect(fill = NA, linewidth = 0.8)) )

####. contact area ####

data_controlled_summary %>% 
  ggplot(aes(x = contact_area, y = areaSmooth_med)) +
  #facet_wrap(~ type, scales = "free", strip.position = "bottom") +
  geom_boxplot(show.legend = FALSE) +
  labs(title = "Contact area", 
       x = "", 
       y = expression("Median contact area (cm"^2*")") ) 


####.  depth ####

data_controlled_summary %>% 
  ggplot(aes(x = force, y = depthSmooth_med)) +
  #facet_wrap(~ type, strip.position = "bottom") +
  geom_boxplot(show.legend = FALSE) +
  labs(title = "Indentation depth", 
       x = "", 
       y = expression("Median depth (cm)") )


####. absolute velocity ####

data_controlled_summary %>% 
  ggplot(aes(x = speed, y = velAbsSmooth_mean, colour = speed)) +
  facet_wrap(~ type, scales = "free", strip.position = "bottom") +
  geom_boxplot(show.legend = FALSE, outlier.shape = 21) +
  labs(title = "Absolute velocity", 
       x = "Contact instruction", 
       y = expression("Median absolute velocity (cm/s)") )


####. lateral velocity ####

data_controlled_summary %>% 
  ggplot(aes(x = speed, y = velLatSmooth_mean_dir, colour = speed)) +
  facet_wrap(~ type, scales = "free", strip.position = "bottom") +
  geom_boxplot(show.legend = FALSE, outlier.shape = 21) +
  labs(title = "Lateral velocity", 
       x = "Contact instruction", 
       y = expression("Median lateral velocity (cm/s)") )


####. longitudinal velocity ####

data_controlled_summary %>% 
  filter(type == "stroke") %>% 
  ggplot(aes(x = speed, y = velLongSmooth_med)) +
  facet_wrap(~ type, scales = "free", strip.position = "bottom") +
  geom_boxplot(show.legend = FALSE, outlier.shape = 21) +
  labs(title = "Longitudinal velocity", 
       x = "Cued (cm/s)", 
       y = expression("Median longitudinal velocity (cm/s)") )

####. vertical velocity ####

data_controlled_summary %>% 
  filter(type == "tap") %>% 
  ggplot(aes(x = speed, y = velVertSmooth_med)) +
  facet_wrap(~ type, scales = "free", strip.position = "bottom") +
  geom_boxplot(show.legend = FALSE, outlier.shape = 21) +
  labs(title = "Vertical velocity", 
       x = "Cued (cm/s)", 
       y = expression("Median longitudinal velocity (cm/s)") )


#### read in expressions data ####
data_files_expressions <- list.files(CONTACT_DATA_FOLDER, full.names = TRUE, recursive = TRUE) %>%
  setdiff(data_files_controlled)
