# Generated by Django 2.2.6 on 2024-03-16 03:55

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0004_auto_20240316_0905'),
    ]

    operations = [
        migrations.RenameField(
            model_name='input_images',
            old_name='Input_image',
            new_name='Input',
        ),
    ]
