# libraries --------------------------------------------------------------------
library(ggplot2)
library(cowplot)
library(data.table)
library(tidyverse)

# constants --------------------------------------------------------------------
electrodes <- c(
  "Fp1",
  "Fp2",
  "F3",
  "F4",
  "C3",
  "C4",
  "P3",
  "P4",
  "O1",
  "O2",
  "F7",
  "F8",
  "T3",
  "T4",
  "T5",
  "T6",
  "Fz",
  "Cz",
  "Pz"
)
left_electrodes <- c("Fp1", "F7", "F3", "T3", "C3", "T5", "P3", "O1")
right_electrodes <- c("Fp2", "F8", "F4", "T4", "C4", "T6", "P4", "O2")


# matrix plot ------------------------------------------------------------------
matrix_plot <- function(filename, title = "", legend = FALSE) {
  # connectivity matrix
  c <- read.table(filename, header = FALSE, sep = ",")

  colnames(c) <- electrodes
  rownames(c) <- electrodes

  # reshape data
  c <- c %>%
    rownames_to_column("Var1") %>%
    gather(Var2, value, -Var1) %>%
    filter(Var1 %in% left_electrodes, Var2 %in% right_electrodes) %>%
    mutate(
      Var1 = factor(Var1, levels = rev(left_electrodes)),
      Var2 = factor(Var2, levels = right_electrodes)
    )

  # plot
  plot <- ggplot(c, aes(Var2, Var1)) +
    geom_tile(aes(fill = value)) +
    scale_fill_gradient2(low = "white", high = "darkblue", guide = "colorbar") +
    theme_minimal() +
    theme(
      axis.ticks.x = element_blank(), axis.ticks.y = element_blank(),
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
      legend.title = element_blank(),
      panel.border = element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      text = element_text(size = 18)
    ) +
    xlab("Right Hemisphere") +
    ylab("Left Hemisphere") +
    ggtitle(title) +
    theme(plot.title = element_text(hjust = 0.5))

  if (!legend) {
    plot <- plot + theme(legend.position = "none")
  }

  plot
}

# connectome plots
p1 <- matrix_plot("./data/connectomes/test/alpha/T003_alpha_task-rest_connectome_eeg.csv", title = "Test patient")
p1
p2 <- matrix_plot("./data/connectomes/control/alpha/C028_alpha_task-rest_connectome_eeg.csv", title = "Control patient")
p2

plot_grid(p1, p2, ncol = 2, scale = 0.9) +
  theme(plot.background = element_rect(fill = "white", color = NA))

ggsave(paste0("./figures/connectomes.png"),
  width = 3840,
  height = 2160,
  dpi = 300,
  units = "px"
)
