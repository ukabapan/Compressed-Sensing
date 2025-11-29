function [stereo_out, Fs] = load_stereo_audio( audio_file_location, truncate_duration)
    arguments
        audio_file_location 
        truncate_duration {mustBeNonnegative} = 0
    end

    if ~exist(audio_file_location, 'file')
            error('File "%s" not found.', audio_file_location);
    end
    
    audio_info = audioinfo(audio_file_location);
    
    if audio_info.NumChannels ~= 2
        error('Input audio must be stereo.');
    end

    if truncate_duration ~= 0
        num_samples = min(audio_info.TotalSamples, truncate_duration * audio_info.SampleRate);
    else
        num_samples = audio_info.TotalSamples;
    end

    [x_raw,Fs] = audioread(audio_file_location,[1, num_samples]); 
    stereo_out = x_raw';
end

