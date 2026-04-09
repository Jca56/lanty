/// Download and prepare Arch Wiki training data.
///
/// This downloads a dump of the Arch Wiki, cleans the markup,
/// and saves plain text files ready for training.
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

fn main() {
    println!("=== Lanty Data Preparation ===");
    println!();

    let data_dir = "data";
    fs::create_dir_all(data_dir).expect("Failed to create data directory");

    // Check if we already have data
    let existing_files: Vec<_> = fs::read_dir(data_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "txt")
                .unwrap_or(false)
        })
        .collect();

    if !existing_files.is_empty() {
        println!(
            "Found {} existing .txt files in data/",
            existing_files.len()
        );
        println!("Delete them first if you want to re-download.");
        println!();
    }

    println!("For the Arch Wiki data, you have two options:");
    println!();
    println!("  OPTION 1: Manual download (recommended for large dataset)");
    println!("    1. Go to: https://wiki.archlinux.org/title/Special:Export");
    println!("    2. Export pages you want (or use the full dump)");
    println!("    3. Save the XML file to data/archwiki.xml");
    println!("    4. Run this tool again to process it");
    println!();
    println!("  OPTION 2: Use sample data to test the pipeline");
    println!("    This will create a small sample dataset so you can");
    println!("    verify everything works before downloading the full wiki.");
    println!();

    // Check for XML dump
    let xml_path = Path::new(data_dir).join("archwiki.xml");
    if xml_path.exists() {
        println!("Found archwiki.xml! Processing...");
        process_wiki_xml(&xml_path, data_dir);
        return;
    }

    // Create sample data for testing
    println!("Creating sample dataset for testing...");
    create_sample_data(data_dir);

    println!();
    println!("Sample data created in data/");
    println!("This is enough to test the full pipeline (tokenizer + model training).");
    println!();
    println!("For a real model, you'll want much more data.");
    println!("See the instructions above for downloading the Arch Wiki.");
}

fn create_sample_data(data_dir: &str) {
    // A substantial sample of Arch Linux knowledge for testing
    let arch_basics = r#"Arch Linux is an independently developed, x86-64 general-purpose GNU/Linux distribution that strives to provide the latest stable versions of most software by following a rolling-release model. The default installation is a minimal base system, configured by the user to only add what is purposely required.

Pacman is the Arch Linux package manager. It combines a simple binary package format with an easy-to-use build system. The goal of pacman is to make it possible to easily manage packages, whether they are from the official repositories or the user's own builds.

To install a package use pacman -S package_name. To update the system use pacman -Syu. To search for a package use pacman -Ss search_term. To remove a package use pacman -R package_name. To remove a package and its dependencies use pacman -Rs package_name.

The Arch User Repository (AUR) is a community-driven repository for Arch Linux users. It contains package descriptions (PKGBUILDs) that allow you to compile a package from source with makepkg and then install it via pacman. The AUR was created to organize and share new packages from the community and to help expedite popular packages' inclusion into the extra repository.

systemd is a system and service manager for Linux, compatible with SysV and LSB init scripts. systemd provides aggressive parallelization capabilities, uses socket and D-Bus activation for starting services, offers on-demand starting of daemons, keeps track of processes using Linux control groups, maintains mount and automount points, and implements an elaborate transactional dependency-based service control logic.

The Linux kernel is the core of the operating system. It manages hardware resources and provides essential services to other programs. The kernel handles process scheduling, memory management, device drivers, file systems, and networking. Arch Linux typically uses the latest stable kernel release, though LTS kernels are also available.

The boot process in Arch Linux typically involves the BIOS or UEFI firmware loading a bootloader such as GRUB or systemd-boot. The bootloader then loads the Linux kernel and initial ramdisk (initramfs) into memory. The kernel initializes hardware, mounts the root filesystem, and starts the init system (systemd).

Filesystem Hierarchy Standard defines the directory structure and directory contents in Unix-like operating systems. The root directory is /. The /etc directory contains system configuration files. The /home directory contains user home directories. The /usr directory contains user utilities and applications. The /var directory contains variable data files.

Network configuration in Arch Linux can be managed through several tools. systemd-networkd is a system daemon that manages network configurations. NetworkManager is a program for providing detection and configuration for systems to automatically connect to networks. iwd (iNet wireless daemon) is a wireless daemon for Linux written by Intel.

Display servers handle the rendering of graphical user interfaces. Xorg is the traditional display server for Linux. Wayland is a newer protocol that aims to replace X11 with a more modern and secure architecture. Compositors like Sway, Hyprland, and wlroots-based compositors implement the Wayland protocol.

The Arch Build System (ABS) is a ports-like system for building and packaging software from source code. While pacman is the Arch tool for binary package management, the ABS is a collection of tools for compiling source into installable .pkg.tar.zst packages.

Configuration files in Arch Linux are typically found in the /etc directory. User-specific configuration is stored in the home directory, often in hidden files or directories starting with a dot. The XDG Base Directory Specification defines where these files should be located.

Security in Arch Linux involves multiple layers. File permissions control access to files and directories using the traditional Unix permission model. Firewalls like iptables and nftables filter network traffic. SELinux and AppArmor provide mandatory access control. SSH keys provide secure authentication for remote access.

Disk management involves partitioning, formatting, and mounting storage devices. Tools like fdisk, gdisk, and parted handle partitioning. File systems like ext4, btrfs, and xfs can be created with mkfs utilities. The /etc/fstab file defines how disk partitions and other block devices should be mounted.

Package building in Arch Linux uses PKGBUILD files. A PKGBUILD describes the package: its name, version, source URL, build instructions, and dependencies. The makepkg tool reads the PKGBUILD and produces a binary package that can be installed with pacman.

Arch Linux uses a rolling release model. This means there are no fixed release versions. Instead, the system is kept up to date by regularly running pacman -Syu. This approach provides the latest software but requires the user to be aware of potential breaking changes.
"#;

    let linux_general = r#"Linux is a family of open-source Unix-like operating systems based on the Linux kernel. The kernel was first released by Linus Torvalds in 1991. Linux is typically packaged as a Linux distribution, which includes the kernel and supporting system software and libraries.

The shell is a command-line interpreter that provides a user interface for the operating system. Bash (Bourne Again Shell) is the most common shell on Linux systems. Zsh is an extended shell with many improvements over bash. Fish is a smart and user-friendly command line shell.

Processes in Linux are instances of running programs. The init system (systemd on Arch) is the first process started by the kernel (PID 1). Each process has a process ID (PID), a parent process, and can spawn child processes. The ps command lists running processes. The top and htop commands provide interactive process monitoring.

File permissions in Linux use a three-tier model: owner, group, and others. Each tier can have read (r), write (w), and execute (x) permissions. The chmod command changes permissions. The chown command changes file ownership. Permissions can be represented numerically (e.g., 755) or symbolically (e.g., rwxr-xr-x).

The Linux filesystem is organized as a single tree rooted at /. Devices, files, and network resources all appear as entries in this tree. Mount points allow different filesystems to be attached at different locations. Virtual filesystems like /proc and /sys expose kernel and hardware information as files.

Environment variables are dynamic named values that affect the running processes. PATH defines where the shell looks for executable files. HOME points to the user's home directory. The export command sets environment variables. The .bashrc or .zshrc files configure the shell environment at login.

Text processing is a fundamental skill in Linux. Commands like grep search for patterns in text. sed performs stream editing. awk is a pattern scanning and processing language. cut, sort, uniq, and wc are commonly used for text manipulation.

Package managers handle the installation, update, and removal of software. APT is used on Debian-based systems. DNF is used on Fedora. Pacman is used on Arch Linux. Each package manager maintains a database of available packages and their dependencies.

Cron jobs and systemd timers schedule recurring tasks. Crontab files define when commands should run. systemd timer units provide more flexible scheduling with calendar expressions and monotonic timers. The at command schedules one-time tasks.

Virtual terminals and terminal emulators provide command-line access. Linux provides several virtual consoles accessible via Ctrl+Alt+F1 through F6. Terminal emulators like Alacritty, Kitty, and foot run within a graphical environment.

SSH (Secure Shell) enables secure remote access to Linux systems. SSH uses public-key cryptography for authentication. The ssh-keygen command generates key pairs. The sshd daemon listens for incoming connections. SSH config files customize connection settings for different hosts.

Containers and virtualization are important modern Linux technologies. Docker provides application containerization. Podman offers a daemonless container engine. QEMU and KVM enable hardware virtualization. systemd-nspawn provides lightweight system containers.
"#;

    let path = Path::new(data_dir);

    fs::write(path.join("arch_linux_basics.txt"), arch_basics).expect("Failed to write sample data");
    println!("  Created: arch_linux_basics.txt ({:.1} KB)", arch_basics.len() as f64 / 1024.0);

    fs::write(path.join("linux_general.txt"), linux_general).expect("Failed to write sample data");
    println!("  Created: linux_general.txt ({:.1} KB)", linux_general.len() as f64 / 1024.0);
}

fn process_wiki_xml(xml_path: &Path, data_dir: &str) {
    let content = fs::read_to_string(xml_path).expect("Failed to read XML file");

    // Simple XML text extraction - pull text content from <text> tags
    let mut articles = Vec::new();
    let mut current_title = String::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("<title>") && trimmed.ends_with("</title>") {
            current_title = trimmed
                .strip_prefix("<title>")
                .unwrap()
                .strip_suffix("</title>")
                .unwrap()
                .to_string();
        }
    }

    // Extract text between <text> and </text> tags
    let mut in_text = false;
    let mut current_text = String::new();
    let mut article_count = 0;

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.contains("<text") {
            in_text = true;
            // Get content after the opening tag
            if let Some(pos) = trimmed.find('>') {
                current_text.push_str(&trimmed[pos + 1..]);
                current_text.push('\n');
            }
            continue;
        }

        if trimmed.contains("</text>") {
            in_text = false;
            if let Some(pos) = trimmed.find("</text>") {
                current_text.push_str(&trimmed[..pos]);
            }

            // Clean wiki markup
            let cleaned = clean_wiki_text(&current_text);
            if cleaned.len() > 200 {
                articles.push(cleaned);
                article_count += 1;
            }
            current_text.clear();
            continue;
        }

        if in_text {
            current_text.push_str(line);
            current_text.push('\n');
        }
    }

    // Write articles to text files (batch them to avoid too many small files)
    let batch_size = 50;
    let mut batch_num = 0;
    let mut batch_text = String::new();
    let mut batch_count = 0;

    for article in &articles {
        batch_text.push_str(article);
        batch_text.push_str("\n\n");
        batch_count += 1;

        if batch_count >= batch_size {
            let filename = format!("{}/archwiki_batch_{:04}.txt", data_dir, batch_num);
            fs::write(&filename, &batch_text).expect("Failed to write batch");
            println!("  Written: {} ({} articles)", filename, batch_count);
            batch_text.clear();
            batch_count = 0;
            batch_num += 1;
        }
    }

    if !batch_text.is_empty() {
        let filename = format!("{}/archwiki_batch_{:04}.txt", data_dir, batch_num);
        fs::write(&filename, &batch_text).expect("Failed to write batch");
        println!("  Written: {} ({} articles)", filename, batch_count);
    }

    println!();
    println!("Processed {} articles from the Arch Wiki dump.", article_count);
}

/// Remove common wiki markup to get clean text
fn clean_wiki_text(text: &str) -> String {
    let mut result = String::new();

    for line in text.lines() {
        let trimmed = line.trim();

        // Skip empty lines, templates, categories
        if trimmed.is_empty()
            || trimmed.starts_with("{{")
            || trimmed.starts_with("}}")
            || trimmed.starts_with("[[Category:")
            || trimmed.starts_with("[[File:")
            || trimmed.starts_with("[[Image:")
            || trimmed.starts_with("{|")
            || trimmed.starts_with("|}")
            || trimmed.starts_with("|-")
            || trimmed.starts_with("| ")
            || trimmed.starts_with("! ")
        {
            continue;
        }

        let mut clean = trimmed.to_string();

        // Remove wiki links but keep the display text: [[target|display]] -> display
        while let Some(start) = clean.find("[[") {
            if let Some(end) = clean[start..].find("]]") {
                let inner = &clean[start + 2..start + end];
                let display = if let Some(pipe) = inner.find('|') {
                    &inner[pipe + 1..]
                } else {
                    inner
                };
                let display = display.to_string();
                clean = format!("{}{}{}", &clean[..start], display, &clean[start + end + 2..]);
            } else {
                break;
            }
        }

        // Remove bold/italic markers
        clean = clean.replace("'''", "").replace("''", "");

        // Remove HTML tags
        let re_html = regex::Regex::new(r"<[^>]+>").unwrap();
        clean = re_html.replace_all(&clean, "").to_string();

        // Remove section headers markup but keep text
        if clean.starts_with("==") && clean.ends_with("==") {
            clean = clean.trim_matches('=').trim().to_string();
        }

        if !clean.is_empty() {
            result.push_str(&clean);
            result.push('\n');
        }
    }

    result
}
