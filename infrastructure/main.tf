terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "6.8.0"
    }
  }
}

provider "google" {
  project = "msgai-projects"
  region  = "us-central1"
  zone    = "us-central1-c"
}



# This code is compatible with Terraform 4.25.0 and versions that are backwards compatible to 4.25.0.
# For information about validating this Terraform code, see https://developer.hashicorp.com/terraform/tutorials/gcp-get-started/google-cloud-platform-build#format-and-validate-the-configuration

resource "google_compute_instance" "unibern-msgai-project" {

  machine_type = "g2-standard-4"
  name         = "unibern-msgai-project"
  tags = ["http-server", "https-server"]
  zone = "us-central1-c"

  boot_disk {
    auto_delete = true
    device_name = "unibern-msgai-project"

    initialize_params {
      image = "projects/ml-images/global/images/c0-deeplearning-common-cu123-v20240922-debian-11"
      size  = 150
      type  = "pd-balanced"
    }

    mode = "READ_WRITE"
  }

  can_ip_forward      = false
  deletion_protection = false
  enable_display      = false

  guest_accelerator {
    count = 1
    type  = "projects/msgai-projects/zones/us-central1-c/acceleratorTypes/nvidia-l4"
  }

  labels = {
    goog-ec-src = "vm_add-tf"
  }


  network_interface {
    access_config {
      network_tier = "PREMIUM"
    }

    queue_count = 0
    stack_type  = "IPV4_ONLY"
    subnetwork  = "projects/msgai-projects/regions/us-central1/subnetworks/default"
  }

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "TERMINATE"
    preemptible         = false
    provisioning_model  = "STANDARD"
  }

  service_account {
    email  = "616549020550-compute@developer.gserviceaccount.com"
    scopes = ["https://www.googleapis.com/auth/devstorage.read_only", "https://www.googleapis.com/auth/logging.write", "https://www.googleapis.com/auth/monitoring.write", "https://www.googleapis.com/auth/service.management.readonly", "https://www.googleapis.com/auth/servicecontrol", "https://www.googleapis.com/auth/trace.append"]
  }

  shielded_instance_config {
    enable_integrity_monitoring = true
    enable_secure_boot          = false
    enable_vtpm                 = true
  }

}

resource "google_compute_disk" "unibern-msgai-project" {
  name = "unibern-msgai-project"
  type = "pd-balanced"
  zone = "us-central1-c"
  size = 150
  depends_on = [ google_compute_instance.unibern-msgai-project ]
}

# resource "google_compute_instance" "unibern-msgai-projects-power" {
#   boot_disk {
#     auto_delete = true
#     device_name = "unibern-msgai-projects-power-disk"

#     initialize_params {
#       image = "projects/ml-images/global/images/c0-deeplearning-common-gpu-v20240922-debian-11-py310"
#       size  = 100
#       type  = "pd-balanced"
#     }

#     mode = "READ_WRITE"
#   }

#   can_ip_forward      = false
#   deletion_protection = false
#   enable_display      = true

#   guest_accelerator {
#     count = 1
#     type  = "projects/msgai-projects/zones/us-central1-a/acceleratorTypes/nvidia-l4"
#   }

#   labels = {
#     goog-ec-src = "vm_add-tf"
#   }

#   machine_type = "g2-standard-4"
#   name         = "unibern-msgai-projects-power"

#   network_interface {
#     access_config {
#       network_tier = "PREMIUM"
#     }

#     queue_count = 0
#     stack_type  = "IPV4_ONLY"
#     subnetwork  = "projects/msgai-projects/regions/us-central1/subnetworks/default"
#   }

#   scheduling {
#     automatic_restart   = true
#     on_host_maintenance = "TERMINATE"
#     preemptible         = false
#     provisioning_model  = "STANDARD"
#   }

#   service_account {
#     email  = "616549020550-compute@developer.gserviceaccount.com"
#     scopes = ["https://www.googleapis.com/auth/devstorage.read_only", "https://www.googleapis.com/auth/logging.write", "https://www.googleapis.com/auth/monitoring.write", "https://www.googleapis.com/auth/service.management.readonly", "https://www.googleapis.com/auth/servicecontrol", "https://www.googleapis.com/auth/trace.append"]
#   }

#   shielded_instance_config {
#     enable_integrity_monitoring = true
#     enable_secure_boot          = false
#     enable_vtpm                 = true
#   }

#   tags = ["http-server", "https-server"]
#   zone = "us-central1-a"
# }
